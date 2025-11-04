from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

config = _config.get_config("pi0_lora_effort_history")
checkpoint_dir = "/home/kelin/TA-VLA/checkpoints/pi0_lora_er_effort_history/ER_test/29999"

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

dataset = LeRobotDataset("org/ER_TAVLA",local_files_only = True)
a = dataset[0]["observation.effort"].reshape(1,6)
b = a.repeat(10, 1)
prediction = []
gt = []

for i in range(600):
    sample = dataset[i]
    for j in range(b.shape[0] - 1):
        b[j] = b[j+1]
    b[-1] = sample["observation.effort"].reshape(1,6)
    example = {
        "images": {
            "cam_high": sample["observation.images.0_femtobolt"],
            "cam_left_wrist": sample["observation.images.1_gemini330"],
            "cam_right_wrist": sample["observation.images.1_gemini330"],
        },
        "state": sample["observation.state"],
        "effort": b,
        "prompt": "DEBUG",
    }
    action_chunk = policy.infer(example)["actions"]
    prediction.append(action_chunk[0,:7])
    gt.append(np.array(sample["action"]))

prediction = np.array(prediction)
gt = np.array(gt)

for i in range(prediction.shape[1]):
    plt.plot(prediction[:, i], label=f'Joint {i+1}')
    plt.plot(gt[:, i], linestyle='--', label=f'GT_Joint {i+1}')

plt.title("Prediction vs GT")
plt.xlabel("Timestep")
plt.ylabel("Action")
plt.legend()
plt.grid(True)

plt.savefig("/home/kelin/effort_plot.png", dpi=300, bbox_inches='tight')
plt.show()
    
