import torch
from PIL import Image
import wandb
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt
import math

# Import necessary classes from your codebase
from torch.utils.data import DataLoader
from configuration import load_yaml_config  # Replace with the actual module name
from llava_finetune.model import LISA_Model        # Replace with the actual module name
from llava_finetune.utils import CustomDataset, collate_fn    # Replace with the actual module name

# Initialize WandB
wandb.init(project="model_inspection", name="Model Output Inspection", config={"config_file": "config.yaml"})

yaml_config = load_yaml_config("config.yaml")

# Define the configuration dictionary
config = {
    "model_llava_seg": {
        "model_path": "models/model_llava_seg.pth",  # Path to the trained model's state_dict
        "test_jsonl": "data/val-v2.jsonl",                     # Path to the test JSONL file
        "image_dir": yaml_config.dataset.image_dir,             # Directory containing test images
        "num_samples": 1,                                      # Number of samples to inspect
        "device": "cuda" if torch.cuda.is_available() else "cpu",  # Device to run the model on
    }
}

def load_model(model_path, device):
    """
    Load the trained LISA_Model from the saved state_dict.
    
    Args:
        model_path (str): Path to the saved model state_dict.
        device (str): Device to load the model on.

    Returns:
        LISA_Model: Loaded model.
    """
    print("Initializing the Model Architecture")
    # Initialize the model architecture
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully.")
    return model

def load_test_data(val_jsonl, image_dir):
    """
    Load the test dataset and create a DataLoader.
    
    Args:
        val_jsonl (str): Path to the test JSONL file.
        image_dir (str): Directory containing test images.

    Returns:
        DataLoader: DataLoader for the test dataset.
    """
    print("Loading Data")
    data_val = CustomDataset(
        json_path=val_jsonl, image_dir=os.path.join(image_dir, "val")
    )
    data_val = DataLoader(data_val, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    print("Data Loaded Successfully")
    return data_val

def visualize_with_matplotlib(samples):
    """
    Visualize samples using Matplotlib with improved caption formatting.
    
    Args:
        samples (list): List of dictionaries containing image and captions.
    """
    num_samples = len(samples)
    cols = 2
    rows = math.ceil(num_samples / cols)
    plt.figure(figsize=(15, 5 * rows))
    
    for idx, sample in enumerate(samples):
        image = sample["image"]
        query = sample["query"]
        generated = sample["generated"]
        
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(image)
        plt.axis("off")
        caption = f"**Query:** {query}\n**Generated Output:** {generated}"
        plt.title(caption, fontsize=12, wrap=True)
    
    plt.tight_layout()
    plt.show()

def visualize_with_wandb(samples):
    """
    Log samples to WandB for enhanced interactive visualization.
    
    Args:
        samples (list): List of dictionaries containing image and captions.
    """
    wandb_images = []
    for sample in samples:
        image = sample["image"]
        query = sample["query"]
        generated = sample["generated"]
        
        # Convert PIL Image to numpy array
        image_np = image.convert("RGB")
        
        # Create a caption using WandB's caption formatting
        caption = f"**Query:** {query}\n**Generated Output:** {generated}"
        
        wandb_image = wandb.Image(image_np, caption=caption)
        wandb_images.append(wandb_image)
    
    wandb.log({"Sample Outputs": wandb_images})

def inspect_model_outputs(model: LISA_Model, data_loader, num_samples):
    """
    Inspect and visualize model outputs alongside input images from the test set.
    
    Args:
        model (LISA_Model): The trained model to inspect.
        data_loader (DataLoader): DataLoader for the test dataset.
        num_samples (int): Number of samples to inspect.
    """
    samples_inspected = 0
    max_samples = num_samples
    collected_samples = []
    
    print(f"Inspecting {max_samples} samples from the test set...")
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Inspecting Test Samples"):
            if samples_inspected >= max_samples:
                break
            
            images = batch["image"]      # List of PIL Images
            queries = batch["queries"]   # List of query strings
            gt_embs = batch["gt_embs"]   # List of torch.Tensor
            sam_embs = batch["sam_embs"] # List of torch.Tensor
            
            # Generate outputs
            generated_texts, _ = model.generate(
                texts=queries,
                images=images,
                pos_mask_embeds=gt_embs,
                neg_mask_embeds=sam_embs,
            )
            
            for i in range(len(generated_texts)):
                if samples_inspected >= max_samples:
                    break
                
                image = images[i]
                query = queries[i]
                generated = generated_texts[i]
                
                collected_samples.append({
                    "image": image,
                    "query": query,
                    "generated": generated
                })
                
                samples_inspected += 1
    
    # Visualize using Matplotlib
    visualize_with_matplotlib(collected_samples)
    
    # Visualize using WandB
    visualize_with_wandb(collected_samples)
    
    print(f"Inspection completed. {samples_inspected} samples were displayed.")

def main():
    for key, inspect_config in config.items():
        print(f"Config: {key}\n")
        
        # Load the trained model
        model = load_model(
            model_path=inspect_config["model_path"],
            device=inspect_config["device"],
        )
        
        # Load the data
        val_loader = load_test_data(
            val_jsonl=inspect_config["test_jsonl"],
            image_dir=inspect_config["image_dir"],
        )
        
        # Inspect model outputs
        inspect_model_outputs(
            model=model,
            data_loader=val_loader,
            num_samples=inspect_config["num_samples"],
        )
    
    wandb.finish()

if __name__ == "__main__":
    main()
