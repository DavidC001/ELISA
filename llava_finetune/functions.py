import torch
from llava_finetune.utils import initialize_wandb
from llava_finetune.model import LISA_Model
from tqdm.auto import tqdm
import os
import wandb

# ==========================
# 6. Run Experiment Function
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
    print(f"========================\nRunning Experiment: {exp_name}\n========================")
    
    # Initialize wandb for this experiment
    initialize_wandb(exp_name, exp_config)
    
    # Unpack data loaders
    data_loader, data_val_loader, data_test_loader = data_loaders
    
    # Load model with experiment-specific parameters
    print("Loading Model")
    model = LISA_Model(config.llava.model, data_loader.dataset[0]["gt_embs"].shape[1], **exp_config.get("model_params", {}))
    model.to("cuda")

    # Set up optimizer with experiment-specific learning rates
    adapter_lr = exp_config["optimizer"]["adapter_lr"]
    lora_lr = exp_config["optimizer"]["lora_lr"]

    adapter_params = [p for p in model.adapter.parameters()]
    lora_params = [p for p in model.llava_model.llava_model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam([
        {"params": adapter_params, "lr": adapter_lr},
        {"params": lora_params, "lr": lora_lr},
    ])

    best_f1 = 0
    print("Starting Training")

    EPOCHS = exp_config["epochs"]
    log_interval = exp_config.get("log_interval", 10)

    for epoch in range(EPOCHS):
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
                optimizer
            )
            losses.append(loss.item())
            if batch_i % log_interval == 0:
                wandb.log({"train/loss": loss.item(), "epoch": epoch+1})

        avg_loss = sum(losses) / len(losses) if losses else 0
        tqdm.write(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_loss:.4f}")
        wandb.log({"train/avg_loss": avg_loss, "epoch": epoch+1})

        # Validation every 10 epochs or last epoch
        if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCHS:
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
                for batch in tqdm(data_val_loader, desc=f"Validation Epoch {epoch+1}", leave=False):
                    out = model.generate(
                        batch["queries"],
                        batch["image"],
                        batch["gt_embs"],
                        batch["sam_embs"],
                    )
                    for i in range(len(out)):
                        generated_masks = [int(m.strip(">")) for m in out[i].split("<SEG_MASK_")[1:] if m.endswith(">")]
                        positive_masks = set(range(1, len(batch["gt_embs"][i]) + 1))
                        negative_masks = set(range(len(batch["gt_embs"][i]) + 1, len(batch["gt_embs"][i]) + len(batch["sam_embs"][i]) + 1))

                        tp = len(set(generated_masks).intersection(positive_masks))
                        fp = len(set(generated_masks).intersection(negative_masks))
                        fn = len(positive_masks.difference(generated_masks))
                        tn = len(negative_masks.difference(generated_masks))

                        accuracy.append((tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0)
                        precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
                        recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                        f1.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0)

                        # Random guess metrics
                        rand_tp = len(positive_masks) / 2
                        rand_fp = len(negative_masks) / 2
                        rand_fn = len(positive_masks) / 2
                        rand_tn = len(negative_masks) / 2

                        rand_accuracy.append((rand_tp + rand_tn) / (rand_tp + rand_tn + rand_fp + rand_fn) if (rand_tp + rand_tn + rand_fp + rand_fn) > 0 else 0)
                        rand_precision.append(rand_tp / (rand_tp + rand_fp) if (rand_tp + rand_fp) > 0 else 0)
                        rand_recall.append(rand_tp / (rand_tp + rand_fn) if (rand_tp + rand_fn) > 0 else 0)
                        rand_f1.append(2 * rand_tp / (2 * rand_tp + rand_fp + rand_fn) if (2 * rand_tp + rand_fp + rand_fn) > 0 else 0)

            # Calculate averages
            accuracy_avg = sum(accuracy) / len(accuracy) if accuracy else 0
            precision_avg = sum(precision) / len(precision) if precision else 0
            recall_avg = sum(recall) / len(recall) if recall else 0
            f1_avg = sum(f1) / len(f1) if f1 else 0
            rand_accuracy_avg = sum(rand_accuracy) / len(rand_accuracy) if rand_accuracy else 0
            rand_precision_avg = sum(rand_precision) / len(rand_precision) if rand_precision else 0
            rand_recall_avg = sum(rand_recall) / len(rand_recall) if rand_recall else 0
            rand_f1_avg = sum(rand_f1) / len(rand_f1) if rand_f1 else 0

            tqdm.write(f"Validation - Epoch {epoch+1}")
            tqdm.write(f"Accuracy: {accuracy_avg:.4f} vs (rand) {rand_accuracy_avg:.4f}")
            tqdm.write(f"Precision: {precision_avg:.4f} vs (rand) {rand_precision_avg:.4f}")
            tqdm.write(f"Recall: {recall_avg:.4f} vs (rand) {rand_recall_avg:.4f}")
            tqdm.write(f"F1: {f1_avg:.4f} vs (rand) {rand_f1_avg:.4f}")

            wandb.log({
                "val/accuracy": accuracy_avg,
                "val/precision": precision_avg,
                "val/recall": recall_avg,
                "val/f1": f1_avg,
                "epoch": epoch+1
            })

            if f1_avg > best_f1:
                best_f1 = f1_avg
                tqdm.write(f"New Best F1: {best_f1:.4f}")
                wandb.log({"best_val_f1": best_f1})

    # ==========================
    # 7. Testing
    # ==========================
    print("Starting Testing")
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
        for batch in tqdm(data_test_loader, desc="Testing", leave=False):
            out = model.generate(
                batch["queries"],
                batch["image"],
                batch["gt_embs"],
                batch["sam_embs"],
            )
            for i in range(len(out)):
                generated_masks = [int(m.strip(">")) for m in out[i].split("<SEG_MASK_")[1:] if m.endswith(">")]
                positive_masks = set(range(1, len(batch["gt_embs"][i]) + 1))
                negative_masks = set(range(len(batch["gt_embs"][i]) + 1, len(batch["gt_embs"][i]) + len(batch["sam_embs"][i]) + 1))

                tp = len(set(generated_masks).intersection(positive_masks))
                fp = len(set(generated_masks).intersection(negative_masks))
                fn = len(positive_masks.difference(generated_masks))
                tn = len(negative_masks.difference(generated_masks))

                accuracy.append((tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0)
                precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
                recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                f1.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0)

                # Random guess metrics
                rand_tp = len(positive_masks) / 2
                rand_fp = len(negative_masks) / 2
                rand_fn = len(positive_masks) / 2
                rand_tn = len(negative_masks) / 2

                rand_accuracy.append((rand_tp + rand_tn) / (rand_tp + rand_tn + rand_fp + rand_fn) if (rand_tp + rand_tn + rand_fp + rand_fn) > 0 else 0)
                rand_precision.append(rand_tp / (rand_tp + rand_fp) if (rand_tp + rand_fp) > 0 else 0)
                rand_recall.append(rand_tp / (rand_tp + rand_fn) if (rand_tp + rand_fn) > 0 else 0)
                rand_f1.append(2 * rand_tp / (2 * rand_tp + rand_fp + rand_fn) if (2 * rand_tp + rand_fp + rand_fn) > 0 else 0)

    # Calculate averages
    accuracy_test = sum(accuracy) / len(accuracy) if accuracy else 0
    precision_test = sum(precision) / len(precision) if precision else 0
    recall_test = sum(recall) / len(recall) if recall else 0
    f1_test = sum(f1) / len(f1) if f1 else 0
    rand_accuracy_avg_test = sum(rand_accuracy) / len(rand_accuracy) if rand_accuracy else 0
    rand_precision_avg_test = sum(rand_precision) / len(rand_precision) if rand_precision else 0
    rand_recall_avg_test = sum(rand_recall) / len(rand_recall) if rand_recall else 0
    rand_f1_avg_test = sum(rand_f1) / len(rand_f1) if rand_f1 else 0

    tqdm.write(f"Test Results:")
    tqdm.write(f"Accuracy: {accuracy_test:.4f} vs (rand) {rand_accuracy_avg_test:.4f}")
    tqdm.write(f"Precision: {precision_test:.4f} vs (rand) {rand_precision_avg_test:.4f}")
    tqdm.write(f"Recall: {recall_test:.4f} vs (rand) {rand_recall_avg_test:.4f}")
    tqdm.write(f"F1: {f1_test:.4f} vs (rand) {rand_f1_avg_test:.4f}")
    tqdm.write(f"Best F1 (Val): {best_f1:.4f}")

    wandb.log({
        "test/accuracy": accuracy_test,
        "test/precision": precision_test,
        "test/recall": recall_test,
        "test/f1": f1_test,
        "best_val_f1": best_f1
    })
    wandb.finish()

    # ==========================
    # 8. Model Saving
    # ==========================
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/model_{exp_name}.pth")
    print(f"Model saved as models/model_{exp_name}.pth")
