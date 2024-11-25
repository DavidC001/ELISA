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
        self.original_vocab_size = model.config.text_config.vocab_size
        self.original_emb_matrix = model.get_input_embeddings().weight.clone().detach()

    def forward(self, input_ids, num_generate=20, **kwargs):
        # Prepare inputs for generation
        inputs = self.llava_model.prepare_inputs_for_generation(input_ids=input_ids, **kwargs, cache_position=torch.tensor([0]))
        inputs["num_logits_to_keep"] = num_generate
        # Forward pass through the model
        outputs = self.llava_model(
            **inputs,
            return_dict=True,
            output_hidden_states=True,
        )
        logits = outputs.logits
        hidden_states = outputs.hidden_states[0]
        # Generate next tokens
        next_tokens = torch.argmax(logits, dim=-1)
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        generated = processor.tokenizer.decode(next_tokens[0])
        print(f"Generated: {generated}")

        return logits

    def add_tokens(self, new_tokens):
        print(f"Adding {new_tokens.size(0)} new tokens to the vocabulary")
        # Get the current embedding layer
        embedding_layer = self.llava_model.get_input_embeddings()
        # Get the current number of tokens and embedding dimension
        old_num_tokens, embedding_dim = embedding_layer.weight.size()
        num_new_tokens = new_tokens.size(0)
        
        # Create a new embedding matrix with the new size
        new_embedding_weights = torch.cat([embedding_layer.weight.clone().detach(), new_tokens], dim=0)

        # Resize token embeddings
        self.llava_model.resize_token_embeddings(old_num_tokens + num_new_tokens)
        
        new_embedding_layer = self.llava_model.get_input_embeddings()
        # Assign the new embedding matrix to the embedding layer
        new_embedding_layer.weight = nn.Parameter(new_embedding_weights)

        # Update the LM head to have the same weights as the embeddings
        output_embedding_layer = self.llava_model.get_output_embeddings()
        output_embedding_layer.weight = new_embedding_layer.weight

        
    def reset_tokens(self):
        print(f"Resetting the token embeddings")
        # Resize token embeddings
        self.llava_model.resize_token_embeddings(self.original_vocab_size)
        # Assign the original embedding matrix to the embedding layer
        embedding_layer = self.llava_model.get_input_embeddings()
        embedding_layer.weight = nn.Parameter(self.original_emb_matrix)
        # Update the LM head to have the same weights as the embeddings
        output_embedding_layer = self.llava_model.get_output_embeddings()
        output_embedding_layer.weight = embedding_layer.weight
        

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
    if 'new_lm_head' not in name and 'lora' not in name and "embed" not in name and "lm_head" not in name:
        param.requires_grad = False

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

vocab_size_general = model.llava_model.config.text_config.vocab_size
emb_size = model.llava_model.config.text_config.hidden_size

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
                processor(text=[responses[i]], images=None, return_tensors="pt", padding=True,)["input_ids"][0][1:]
            )
        
        max_len = max([label.shape[-1] for label in labels])
        new_tokens_1 = torch.zeros((3, emb_size)).to("cuda")
        new_tokens_1.requires_grad = True
        new_tokens_2 = torch.zeros((3, emb_size)).to("cuda")
        new_tokens_2.requires_grad = True
        new_tokens = new_tokens_1 + 2 * new_tokens_2
        
        model.add_tokens(new_tokens)
        
        outputs = model(**inputs, num_generate=max_len+1)
        
        vocab_size = vocab_size_general + 3
        
        # mask out padding tokens
        mask = torch.ones_like(outputs)
        for i, label in enumerate(labels):
            mask[i, label.shape[-1]:, :] = 0
            
        outputs = outputs * mask
        
        # stack labels by padding at the beginning of the sequence
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=processor.tokenizer.pad_token_id, padding_side="left")
        # convert to 1 - hot encoding for loss calculation
        labels = nn.functional.one_hot(labels, num_classes=vocab_size).float().to("cuda")
        
        # Calculate loss ignoring padding tokens
        loss = nn.functional.cross_entropy(outputs[:,:-1,:].view(-1, vocab_size), labels.view(-1, vocab_size))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # copy gradients to new tokens
        new_tokens.backward(model.llava_model.get_input_embeddings().weight.grad[-new_tokens.size(0):])
        
        # check if new tokens have gradients
        print(f"New tokens gradients: {new_tokens.grad}")
        breakpoint()
        
        optimizer.step()
        
        # Reset token embeddings
        model.reset_tokens()
        processor.tokenizer.reset_tokenizer()
        
        print(f"Loss: {loss.item()}")