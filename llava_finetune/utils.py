import torch
import json
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import wandb

# ==========================
# 1. Dataset Definition
# ==========================
class CustomDataset(Dataset):
    def __init__(self, json_path, image_dir, exp_json_path=None):
        self.image_dir = image_dir
        self.data = self.load_data(json_path, image_dir, exp_json_path)

    def load_data(self, json_path, image_dir, exp_json_path):
        data = []
        answers = {}
        if exp_json_path:
            with open(exp_json_path, "r") as f:
                exp_data = json.load(f)
            for sample in exp_data:
                image = sample["image"]
                answer = sample["outputs"]
                answers[image] = answer

        with open(json_path, "r") as f:
            for line in f:
                sample = json.loads(line)
                image = sample["img"]
                image_json_path = os.path.join(image_dir, image.split(".")[0] + ".json")
                with open(image_json_path, "r") as f2:
                    image_queries = json.load(f2)["text"]

                if (len(sample["gt_embs"]) == 0) or (len(sample["sam_embs"]) == 0):
                    continue

                data.append({
                    "image": Image.open(os.path.join(image_dir, image)),
                    "queries": image_queries,
                    "answer": answers.get(image, None),
                    "gt_embs": torch.tensor(sample["gt_embs"]).view(len(sample["gt_embs"]), -1),
                    "sam_embs": torch.tensor(sample["sam_embs"]).view(len(sample["sam_embs"]), -1),
                })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample["image"]
        query = sample["queries"][torch.randint(0, len(sample["queries"]), (1,)).item()]
        return {
            "image": image,
            "queries": query,
            "answer": sample["answer"],
            "gt_embs": sample["gt_embs"],
            "sam_embs": sample["sam_embs"],
        }

def collate_fn(batch):
    new_batch = {}
    for key in batch[0]:
        new_batch[key] = [sample[key] for sample in batch]
    return new_batch

def get_dataloaders(explanatory_train, image_dir, batch_size, train_jsonl, val_jsonl, test_jsonl):
    """Get the training, validation, and test data loaders.

    Args:
        explanatory_train (_type_): file path to the explanatory data for training
        image_dir (_type_): path to the directory containing the images
        batch_size (_type_): batch size for the data loaders
        train_jsonl (_type_): jsonl file containing the training data masks
        val_jsonl (_type_): jsonl file containing the validation data masks
        test_jsonl (_type_): jsonl file containing the test data masks

    Returns:
        DataLoader: training data loader
        DataLoader: validation data loader
        DataLoader: test data loader
    """
    print("Loading Training Data")
    data_train = CustomDataset(
        json_path=train_jsonl,
        image_dir=os.path.join(image_dir, "train"),
        exp_json_path=explanatory_train
    )

    print("Loading Validation Data")
    data_val = CustomDataset(
        json_path=val_jsonl,
        image_dir=os.path.join(image_dir, "val")
    )

    print("Loading Test Data")
    data_test = CustomDataset(
        json_path=test_jsonl,
        image_dir=os.path.join(image_dir, "test")
    )

    # Create DataLoaders
    data_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    data_val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    data_test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return data_loader, data_val_loader, data_test_loader
    
# ==========================
# 2. WandB Initialization
# ==========================
def initialize_wandb(exp_name, exp_config):
    """
    Initialize a wandb run for the experiment.

    Args:
        exp_name (str): Name of the experiment.
        exp_config (dict): Experiment-specific configuration.
    """
    wandb.init(
        project="LISA_ACV",
        name=exp_name+"_"+wandb.util.generate_id(),
        config=exp_config
    )
