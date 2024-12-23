import warnings
import torch
import random
import numpy as np

import wandb
from configuration import load_yaml_config
from llava_finetune.functions import run_experiment
from llava_finetune.utils import get_dataloaders

warnings.filterwarnings("ignore")

# set seeds for reproducibility (where possible)
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

config = load_yaml_config("config_davide.yaml")
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
    # Load datasets based on configuration
    print("Loading Datasets")
    data_loader, data_val_loader, data_test_loader = get_dataloaders(
        config.dataset.ReasonSeg.json_path,
        config.dataset.ReasonSeg.image_dir,
        2,
        "data/train_final.jsonl",
        "data/val_final.jsonl",
        "data/test_final.jsonl",
    )
    print("Datasets Loaded Successfully")

    # Iterate through each experiment and run it
    for exp_name, exp_config in experiments_config.items():
        run_experiment(
            exp_name=exp_name,
            exp_config=exp_config,
            config=config,
            data_loaders=(data_loader, data_val_loader, data_test_loader),
        )
