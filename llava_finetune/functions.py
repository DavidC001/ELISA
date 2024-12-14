from copy import deepcopy
import json
import torch
from llava_finetune.utils import initialize_wandb
from llava_finetune.model import LISA_Model
from tqdm.auto import tqdm
import os
import wandb
from bitsandbytes.optim import AdamW8bit

# ==========================
# 1. Train step
# ==========================
def train_step(model, data_loader, optimizer, epoch, EPOCHS, log_interval):
    """
    Train the model for one epoch.

    Args:
        model (LISA_Model): The model to train.
        data_loader (DataLoader): The DataLoader for the training data.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        epoch (int): The current epoch number.
        EPOCHS (int): The total number of epochs.
        log_interval (int): How often to log to wandb.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    losses = []
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    for batch_i, batch in enumerate(pbar):
        _, loss = model.optim_step(
            batch["queries"],
            batch["image"],
            batch["answer"],
            batch["gt_embs"],
            batch["sam_embs"],
            optimizer,
        )
        losses.append(loss.item())
        if batch_i % log_interval == 0:
            wandb.log({"train/loss": loss.item(), "epoch": epoch + 1})

    avg_loss = sum(losses) / len(losses) if losses else 0
    return avg_loss


# ==========================
# 2. Validation step
# ==========================


def val_step(model, data_val_loader, epoch):
    """
    Run the validation step for the model.

    Args:
        model (LISA_Model): The model to validate.
        data_val_loader (DataLoader): The DataLoader for the validation data.
        epoch (int): The current epoch number.

    Returns:
        tuple: A tuple containing the average accuracy, precision, recall, F1, and random guess metrics.
    """
    model.eval()
    accuracy = []
    rand_accuracy = []
    precision = []
    rand_precision = []
    recall = []
    rand_recall = []
    f1 = []
    rand_f1 = []

    with torch.no_grad():
        for batch in tqdm(
            data_val_loader, desc=f"Validation Epoch {epoch+1}", leave=False
        ):
            texts, tokens = model.generate(
                [q+" Output the segmentation mask for the object in the image" for q in batch["queries"]],
                batch["image"],
                batch["gt_embs"],
                batch["sam_embs"],
            )
            for i in range(len(tokens)):
                # extract all tokens >model.llava_model.tokenizer_vocab_size to get the generated masks
                generated_masks = set(
                    (
                        tokens[i][tokens[i] > model.llava_model.tokenizer_vocab_size]
                        - model.llava_model.tokenizer_vocab_size
                    ).tolist()
                )
                positive_masks = set(range(1, len(batch["gt_embs"][i]) + 1))
                negative_masks = set(
                    range(
                        len(batch["gt_embs"][i]) + 1,
                        len(batch["gt_embs"][i]) + len(batch["sam_embs"][i]) + 1,
                    )
                )

                tp = len(generated_masks.intersection(positive_masks))
                fp = len(generated_masks.intersection(negative_masks))
                fn = len(positive_masks.difference(generated_masks))
                tn = len(negative_masks.difference(generated_masks))

                accuracy.append(
                    (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                )
                precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
                recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                f1.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0)

                # Random guess metrics
                rand_tp = len(positive_masks) / 2
                rand_fp = len(negative_masks) / 2
                rand_fn = len(positive_masks) / 2
                rand_tn = len(negative_masks) / 2

                rand_accuracy.append(
                    (rand_tp + rand_tn) / (rand_tp + rand_tn + rand_fp + rand_fn)
                    if (rand_tp + rand_tn + rand_fp + rand_fn) > 0
                    else 0
                )
                rand_precision.append(
                    rand_tp / (rand_tp + rand_fp) if (rand_tp + rand_fp) > 0 else 0
                )
                rand_recall.append(
                    rand_tp / (rand_tp + rand_fn) if (rand_tp + rand_fn) > 0 else 0
                )
                rand_f1.append(
                    2 * rand_tp / (2 * rand_tp + rand_fp + rand_fn)
                    if (2 * rand_tp + rand_fp + rand_fn) > 0
                    else 0
                )

    # Calculate averages
    accuracy_avg = sum(accuracy) / len(accuracy) if accuracy else 0
    precision_avg = sum(precision) / len(precision) if precision else 0
    recall_avg = sum(recall) / len(recall) if recall else 0
    f1_avg = sum(f1) / len(f1) if f1 else 0
    rand_accuracy_avg = sum(rand_accuracy) / len(rand_accuracy) if rand_accuracy else 0
    rand_precision_avg = (
        sum(rand_precision) / len(rand_precision) if rand_precision else 0
    )
    rand_recall_avg = sum(rand_recall) / len(rand_recall) if rand_recall else 0
    rand_f1_avg = sum(rand_f1) / len(rand_f1) if rand_f1 else 0

    return (
        accuracy_avg,
        precision_avg,
        recall_avg,
        f1_avg,
        rand_accuracy_avg,
        rand_precision_avg,
        rand_recall_avg,
        rand_f1_avg,
    )


# ==========================
# 3. Run Experiment Function
# ==========================
def run_experiment(exp_name, exp_config, config, data_loaders):
    """
    Executes the training, validation, and testing pipeline for a given experiment.

    Args:
        exp_name (str): Name of the experiment.
        exp_config (dict): Configuration dictionary for the experiment.
        config (object): Global configuration object loaded from YAML.
        data_loaders (tuple): Tuple containing training, validation, and test DataLoaders.
    """
    print(
        f"========================\nRunning Experiment: {exp_name}\n========================"
    )
    os.makedirs("models", exist_ok=True)

    # Initialize wandb for this experiment
    initialize_wandb(exp_name, exp_config)

    # Unpack data loaders
    data_loader, data_val_loader, data_test_loader = data_loaders

    # Load model with experiment-specific parameters
    print("Loading Model")
    model = LISA_Model(
        model_name=config.llava.model,
        seg_emb_size=data_loader.dataset[0]["gt_embs"].shape[1],
        **exp_config.get("model_params", {}),
    )
    model.to("cuda")
    # save model params in json file
    with open(f"models/{exp_name}.json", "w") as f:
        json.dump({"model_name": config.llava.model, "seg_emb_size": data_loader.dataset[0]["gt_embs"].shape[1], **exp_config.get("model_params", {})}, f)
    # save preprocess params in json file
    with open(f"models/preprocess_{exp_name}.json", "w") as f:
        json.dump(exp_config.get("preprocess_params", {}), f)
    
    def verify_models_equality(model1, model2):
        # 1. Compare parameters
        for (name1, p1), (name2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            if name1 != name2:
                print(f"Parameter name mismatch: {name1} vs {name2}")
                return False
            if not torch.equal(p1.data, p2.data):
                print(f"Parameter value mismatch in {name1}")
                print(f"Max difference: {(p1.data - p2.data).abs().max()}")
                return False
        
        # 2. test the model outputs using the same input
        out = val_step(model1, data_val_loader, 0)
        out2 = val_step(model2, data_val_loader, 0)
        
        for i, (o1, o2) in enumerate(zip(out, out2)):
            if o1 > o2 + 1e-6 or o1 < o2 - 1e-6:
                print(f"Output mismatch in {i}")
                print(f"Values: {o1} vs {o2}")
                return False
        
        return True

    def save_state_dict(model, path):
        # Get state dict and ensure it's on CPU
        state_dict = model.state_dict()
        # Remove quantization-specific keys
        cleaned_state = {k: v for k, v in state_dict.items() 
                        if not any(x in k for x in ['.absmax', '.quant_map', '.quant_state'])}
        torch.save(cleaned_state, path)

    # Set up optimizer with experiment-specific learning rates
    adapter_lr = exp_config["optimizer"]["adapter_lr"]
    lora_lr = exp_config["optimizer"]["lora_lr"]

    adapter_params = [p for p in model.adapter.parameters()]
    lora_params = [
        p for p in model.llava_model.llava_model.parameters() if p.requires_grad
    ]

    optimizer = AdamW8bit(
        [
            {"params": adapter_params, "lr": adapter_lr},
            {"params": lora_params, "lr": lora_lr},
        ]
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=exp_config["epochs"], eta_min=exp_config["scheduler"]["eta_min"])

    best_f1 = 0
    print("Starting Training")

    EPOCHS = exp_config["epochs"]
    SKIP_VAL_TEST = exp_config.get("skip_test_val", False)
    log_interval = exp_config.get("log_interval", 10)
    val_every = exp_config.get("val_every", 10)

    for epoch in tqdm(range(EPOCHS)):
        avg_loss = train_step(
            model, data_loader, optimizer, epoch, EPOCHS, log_interval
        )
        tqdm.write(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_loss:.4f}")
        wandb.log({"train/avg_loss": avg_loss, "train/adapter_lr": optimizer.param_groups[0]["lr"], "train/lora_lr": optimizer.param_groups[1]["lr"], "epoch": epoch + 1})

        scheduler.step()

        # Validation every val_every epochs or last epoch
        if (epoch + 1) % val_every == 0 or (epoch + 1) == EPOCHS:
            if not SKIP_VAL_TEST:
                (
                    accuracy_avg,
                    precision_avg,
                    recall_avg,
                    f1_avg,
                    rand_accuracy_avg,
                    rand_precision_avg,
                    rand_recall_avg,
                    rand_f1_avg,
                ) = val_step(model, data_val_loader, epoch)

                tqdm.write(f"Validation - Epoch {epoch+1}")
                tqdm.write(
                    f"Accuracy: {accuracy_avg:.4f} vs (rand) {rand_accuracy_avg:.4f}"
                )
                tqdm.write(
                    f"Precision: {precision_avg:.4f} vs (rand) {rand_precision_avg:.4f}"
                )
                tqdm.write(f"Recall: {recall_avg:.4f} vs (rand) {rand_recall_avg:.4f}")
                tqdm.write(f"F1: {f1_avg:.4f} vs (rand) {rand_f1_avg:.4f}")

                wandb.log(
                    {
                        "val/accuracy": accuracy_avg,
                        "val/precision": precision_avg,
                        "val/recall": recall_avg,
                        "val/f1": f1_avg,
                        "epoch": epoch + 1,
                    }
                )

                if f1_avg > best_f1:
                    best_f1 = f1_avg
                    tqdm.write(f"New Best F1: {best_f1:.4f}")
                    wandb.log({"best_val_f1": best_f1})
                    save_state_dict(model, f"models/{exp_name}.pth")
                
    if  not os.path.exists(f"models/{exp_name}.pth") or SKIP_VAL_TEST:
        print("Best model not found, using the last model for testing")
        save_state_dict(model, f"models/{exp_name}.pth")
    
    # ==========================
    # 7. Testing
    # ==========================
    if not SKIP_VAL_TEST:
        model.load_state_dict(torch.load(f"models/{exp_name}.pth"))

        print("Starting Testing")
        (
            accuracy_test,
            precision_test,
            recall_test,
            f1_test,
            rand_accuracy_avg_test,
            rand_precision_avg_test,
            rand_recall_avg_test,
            rand_f1_avg_test,
        ) = val_step(model, data_test_loader, EPOCHS)

        tqdm.write(f"Test Results:")
        tqdm.write(f"Accuracy: {accuracy_test:.4f} vs (rand) {rand_accuracy_avg_test:.4f}")
        tqdm.write(
            f"Precision: {precision_test:.4f} vs (rand) {rand_precision_avg_test:.4f}"
        )
        tqdm.write(f"Recall: {recall_test:.4f} vs (rand) {rand_recall_avg_test:.4f}")
        tqdm.write(f"F1: {f1_test:.4f} vs (rand) {rand_f1_avg_test:.4f}")
        tqdm.write(f"Best F1 (Val): {best_f1:.4f}")

        wandb.log(
            {
                "test/accuracy": accuracy_test,
                "test/precision": precision_test,
                "test/recall": recall_test,
                "test/f1": f1_test,
                "best_val_f1": best_f1,
            }
        )
    wandb.finish()
    print("Experiment Completed Successfully")


# ==========================
# 4. Other Utility Functions
# ==========================
def load_model(model_path, model_params, device):
    """
    Load the trained LISA_Model from the saved state_dict.

    Args:
        model_path (str): Path to the saved model state_dict.
        model_params (str): Path to the model parameters JSON file.
        device (str): Device to load the model on.

    Returns:
        LISA_Model: Loaded model.
    """
    # check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    print("Initializing the Model Architecture")
    # Initialize the model architecture
    with open(model_params, "r") as file:
        model_params = json.load(file)

    model = LISA_Model(**model_params)
    model.load_state_dict(torch.load(model_path), strict=False)
    model = model.to(device)
    model.eval()

    print("Model loaded successfully.")
    return model