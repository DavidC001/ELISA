import json
import os

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import LlavaForConditionalGeneration, LlavaProcessor, PreTrainedModel, BitsAndBytesConfig
import bitsandbytes as bnb

from configuration import load_yaml_config

from model import CustomModel


def get_model(model_name: str, device: str = "cpu", load_in_4bit: bool = True, load_in_8bit: bool = False)-> CustomModel:
    # Initialize the processor
    processor = LlavaProcessor.from_pretrained(model_name)

    new_tokens = [f"<new_token_{i}>" for i in range(1, 10)]
    processor.tokenizer.add_tokens(new_tokens)

    # Configuration for quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )

    # Load the pre-trained LLava model and wrap it with the custom model
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config
    )

    # Apply LoRA to the LLava model
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Adjust based on the actual module names
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    model = CustomModel(model, processor).to(device)
    
    return model
    
def get_optimizer()

def val_loop():
    pass

def train_loop():
    pass