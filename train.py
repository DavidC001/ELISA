import torch
import warnings
warnings.filterwarnings("ignore")
from llava_finetune.utils import get_dataloaders
from llava_finetune.functions import run_experiment

from configuration import load_yaml_config

config = load_yaml_config("config.yaml")

# ==========================
# 1. Experiment Configurations
# ==========================
experiments_config = {
    "test": {
        "epochs": 200,
        "optimizer": {
            "adapter_lr": 1e-3,
            "lora_lr": 1e-8,
        },
        "model_params": {
            # General Parameters
            "end_turn_token": "<end_of_turn>\n",
            "q4": True, 
            "q8": False, 
            # Adapter Parameters
            "num_linears": 25, 
            "hidden_dim": None, 
            "expand_factor": 2, 
            "dropout": 0.25, 
            "num_heads": 1, 
            "num_queries": 10,
        },
        "log_interval": 10  # How often to log to wandb
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
        5,
        "ReasonSeg/data/train-v2.jsonl",
        "ReasonSeg/data/val-v2.jsonl",
        "ReasonSeg/data/test-v2.jsonl"
    )
    print("Datasets Loaded Successfully")

    # Iterate through each experiment and run it
    for exp_name, exp_config in experiments_config.items():
        run_experiment(
            exp_name=exp_name,
            exp_config=exp_config,
            config=config,
            data_loaders=(data_loader, data_val_loader, data_test_loader)
        )
