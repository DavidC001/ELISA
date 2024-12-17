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
experiments_config = {
    "TEST_REDUCT_NEG": {
        "epochs": 50,
        "skip_test_val": False,
        "optimizer": {
            "adapter_lr": 1e-3,
            "lora_lr": 1e-6,
        },
        "scheduler": {
            "eta_min": 1e-4,
        },
        
        "preprocess_params": {
            "model": "alpha-clip",
            "only_masks": False,
        },
        "model_params": {
            # Model Parameters
            "lora_rank": 16,
            "q4": True,
            "q8": False,
            
            # Adapter Parameters
            "expand_factor": 2,
            "mlp_adapter": True,
            
            # Training Parameters
            "mask_prob": 0.2, # Probability of masking a token in the input
            "dropout": 0.2,
            "noise_level": 1e-6,
            "pos_weight": 2, # Weight for positive samples classes
            "neg_weight": 1, # Weight for negative samples classes
            "temperature": 0.1, # Temperature for logits scaling during training
            
            # Other Parameters
            "end_turn_token": "<end_of_turn>\n",
            "seg_pos": "randomized",  # "before" or "after" the gt answer from the dataset
            "text": True, # Whether to use text or not (add "in the image the objects are ...")
        },
        "log_interval": 10,  # How often to log to wandb
        "val_every": 50,  # How often to run validation
    }
}


# ==========================
# 2. Main Execution
# ==========================
if __name__ == "__main__":
    # Load datasets based on configuration
    print("Loading Datasets")
    data_loader, data_val_loader, data_test_loader = get_dataloaders(
        config.dataset.json_path,
        config.dataset.image_dir,
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
