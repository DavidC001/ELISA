import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, LlavaProcessor, LlavaForConditionalGeneration
from peft import LoraConfig, get_peft_model
import os
import json
from PIL import Image

# Define the custom model class
class CustomModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.llava_model = model
        # Define a new LM head
        self.new_lm_head = nn.Linear(model.config.text_config.hidden_size, 10)

    def forward(self, input_ids, num_generate=20, **kwargs):
        # Get outputs from the base model, including hidden states        
        inputs = self.llava_model.prepare_inputs_for_generation(input_ids=input_ids, **kwargs, cache_position=torch.tensor([0]))
        inputs["num_logits_to_keep"] = num_generate
        outputs = self.llava_model.forward(
            **inputs,
            return_dict=True,
            output_hidden_states=True,
        )
        logits = outputs.logits
            
        hidden_states = outputs.hidden_states[0]
            
        next_tokens = torch.argmax(logits, dim=-1)
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        generated = processor.tokenizer.decode(next_tokens[0])
        print(f"Generated: {generated}")
        
        return logits

model_name = "Intel/llava-gemma-2b"

# Initialize the processor
processor = LlavaProcessor.from_pretrained(model_name)

# Load the pre-trained LLava model and wrap it with the custom model
model = LlavaForConditionalGeneration.from_pretrained(model_name)

# Apply LoRA to the LLava model
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Adjust based on the actual module names
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

model = CustomModel(model).to("cuda")

# Freeze LLava model parameters except for LoRA parameters
for name, param in model.named_parameters():
    if 'new_lm_head' not in name and 'lora' not in name:
        param.requires_grad = False

# Ensure new LM head parameters are trainable
for param in model.new_lm_head.parameters():
    param.requires_grad = True

# Define the dataset class (as provided)
class CustomDataset(Dataset):
    def __init__(self, json_path, image_dir, processor):
        self.image_dir = image_dir
        self.processor = processor
        self.data = self.load_data(json_path)

    def load_data(self, json_path):
        with open(json_path, 'r') as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.data[idx]['image'])
        image = Image.open(image_path).convert("RGB")
        # conversation = [
        #     {

        #     "role": "user",
        #     "content": [
        #         {"type": "text", "text": self.data[idx]['query']},
        #         {"type": "image"},
        #         ],
        #     },
        # ]   
        # query = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        response = self.data[idx]['outputs']
        query = processor.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': f"<image>\n{self.data[idx]['query']}"}, {'role': 'assistant', 'content': response}],
            tokenize=False,
            add_generation_prompt=False
        )
        response = f"{response}<end_of_turn>"
        
        return {
            "image": image,
            "query": query,
            "response": response,
        }

def collate_fn(batch):
    images = [item['image'] for item in batch]
    queries = [item['query'] for item in batch]
    responses = [item['response'] for item in batch]
    return images, queries, responses

json_path = 'C:/Users/david/Documents/progetto/LISA/ReasonSeg/explanatory/train.json'
image_dir = 'C:/Users/david/Documents/progetto/LISA/ReasonSeg/train/'
# Example usage of the dataset and dataloader
dataset = CustomDataset(json_path=json_path, image_dir=image_dir, processor=processor)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

num_epochs = 5

# Prepare optimizer (only trainable parameters)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# Training loop (simplified)
for epoch in range(num_epochs):
    for images, queries, responses in dataloader:
        # Process inputs
        inputs = processor(
            images=images,
            text=queries,
            return_tensors="pt",
            padding=True,
        ).to("cuda") # This pads at the begginning of the sequence so WHOHO we can work with that
        
        # Prepare labels
        labels = []
        for i in range(len(responses)):
            labels.append(
                processor(
                    text=responses, images=None,
                    return_tensors="pt",
                    padding=True,
                )["input_ids"][1:]
            )
        
        outputs = model(**inputs, num_generate=max([len(response.split()) for response in responses])+1)
        
        # breakpoint()
        # loss = outputs.loss
        
        # # Backward pass and optimization
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()
