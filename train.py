import warnings
import torch
import random
import numpy as np

import wandb
from configuration import load_yaml_config
from llava_finetune.functions import run_experiment

warnings.filterwarnings("ignore")

# set seeds for reproducibility (where possible)
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

config = load_yaml_config("config.yaml")
wan_db_token = config.others.wandb_token
wandb.login(key=wan_db_token)

# ==========================
# 1. Experiment Configurations
# ==========================
experiments_config = config.train.experiments


# ==========================
# 2. Main Execution
# ==========================
if __name__ == "__main__":
    # Iterate through each experiment and run it
    for exp_name, exp_config in experiments_config.items():
        run_experiment(
            exp_name=exp_name,
            exp_config=exp_config,
            config=config,
        )
