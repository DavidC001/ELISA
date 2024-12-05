from llava_finetune.model import LISA_Model
import torch
import json
import os
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from configuration import load_yaml_config

config = load_yaml_config("config.yaml")

# Define the dataset class (as provided)
class CustomDataset(Dataset):
    def __init__(self, json_path, image_dir, exp_json_path):
        self.image_dir = image_dir
        self.data = self.load_data(json_path, image_dir, exp_json_path)

    def load_data(self, json_path, image_dir, exp_json_path):
        data = []
        answers = {}
        # Load the explanatory file
        with open(exp_json_path, "r") as f:
            exp_data = json.load(f)
        # Parse the json file to get the answers
        for sample in exp_data:
            image = sample["image"]
            answer = sample["outputs"]
            answers[image] = answer
            
        with open(json_path, "r") as f:
            for line in f:
                sample = json.loads(line)
                image = sample["img"]
                # get the image.json file by removing the .jpg extension and adding .json extension
                image_json_path = os.path.join(image_dir, image.split(".")[0] + ".json")
                with open(image_json_path, "r") as f:
                    image_queries = json.load(f)["text"]
                
                # get the query from the explanatory file
                data.append({
                    "image": image,
                    "queries": image_queries,
                    "answer": answers[image],
                    "gt_embs": torch.tensor(sample["gt_embs"]).view(len(sample["gt_embs"]), -1),
                    "sam_embs": torch.tensor(sample["sam_embs"]).view(len(sample["sam_embs"]), -1),
                })
        return data
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = Image.open(os.path.join(self.image_dir, sample["image"]))
        # pick a random query
        query = sample["queries"][torch.randint(0, len(sample["queries"]), (1,)).item()]
        return {
            "image": image,
            "queries": query,
            "answer": sample["answer"],
            "gt_embs": sample["gt_embs"],
            "sam_embs": sample["sam_embs"],
        }
        
def __collate_fn(batch):
    new_batch = {}
    for key in batch[0]:
        new_batch[key] = [sample[key] for sample in batch]
    return new_batch
    
data = CustomDataset("ReasonSeg/res.jsonl", "ReasonSeg/train", "ReasonSeg/explanatory/train.json")
data_loader = torch.utils.data.DataLoader(data, batch_size=10, shuffle=True, collate_fn=__collate_fn)

# Load the model
model = LISA_Model(config.llava.model, 512)
model.to("cuda")

EPOCHS = 2

adapter_params = [p for p in model.adapter.parameters()]
lora_params = [p for p in model.llava_model.parameters() if p.requires_grad]

optimizer = torch.optim.Adam([
    {"params": adapter_params, "lr": 1e-3},
    {"params": lora_params, "lr": 1e-5},
])

model.train()
for epoch in tqdm(range(EPOCHS)):
    for batch in tqdm(data_loader, desc=f"Epoch {epoch}"):
        print()
        _, loss = model.optim_step(
            batch["queries"], 
            batch["image"], 
            batch["answer"],
            batch["gt_embs"],
            batch["sam_embs"],
            optimizer
        )
        print()
        # set description to the loss
        tqdm.write(f"Loss: {loss.item()}")
    