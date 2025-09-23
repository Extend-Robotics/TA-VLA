import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "s3://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        missing_regex = ".*lora.*" if "effort_proj_in" not in params else ".*lora.*|effort_proj.*"
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=missing_regex)


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    
    rng = np.random.RandomState(42)

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            if v.shape == flat_ref[k].shape:
                result[k] = v.astype(flat_ref[k].dtype)
            elif any(name in k for name in ["action_in_proj", "action_out_proj"]):
                ref_shape = flat_ref[k].shape
                ref_dtype = flat_ref[k].dtype
                loaded = v
                logger.info(f"Reshaping weights for {k}: loaded {loaded.shape} -> reference {ref_shape}")
                
                # kernel (weights)
                if len(ref_shape) == 2 and len(loaded.shape) == 2:
                    fan_in = ref_shape[0]
                    scale = np.sqrt(2.0 / fan_in) * 0.01
                    new_array = rng.normal(0, scale, ref_shape).astype(ref_dtype)
                    new_array[:loaded.shape[0], :loaded.shape[1]] = loaded[:loaded.shape[0], :loaded.shape[1]].astype(ref_dtype)
                    result[k] = new_array
                # bias
                elif len(ref_shape) == 1 and len(loaded.shape) == 1:
                    scale = 0.001
                    new_array = rng.normal(0, scale, ref_shape).astype(ref_dtype)
                    new_array[:loaded.shape[0]] = loaded[:loaded.shape[0]].astype(ref_dtype)
                    result[k] = new_array
                else:
                    raise ValueError
            else:
                raise ValueError(f"Shape mismatch for {k}: loaded {v.shape} vs reference {flat_ref[k].shape}")

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
