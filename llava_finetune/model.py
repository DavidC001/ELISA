import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import (
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    PreTrainedModel,
)

DEBUG_PRINTS = False


class QueryBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_queries,
        expand_factor,
        num_heads,
        dropout,
    ):
        """
        Adapter module from the output of AlphaClip from the segmentation model to the input of the LLava model

        Args:
            input_dim (int): Dimension of the hidden layers in the adapter module
            output_dim (int): Dimension of the output layer in the adapter module
            num_queries (int): Number of queries to use in the multi-head attention layer
            expand_factor (int): Factor to expand the hidden dimension in the FFN
            dropout (float): Dropout rate to apply in the adapter module
            num_heads (int): Number of heads to use in the multi-head attention layer
        """
        super(QueryBlock, self).__init__()

        self.attention = nn.MultiheadAttention(
            input_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )

        query = torch.randn(1, num_queries, input_dim)
        self.query = nn.Parameter(query)

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * expand_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * expand_factor, input_dim),
        )

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, input):
        """
        Forward pass through the adapter module

        Args:
            input (torch.Tensor): Input tensor with shape (num_segments, input_segment_dim)

        Returns:
            x (torch.Tensor): Output tensor with shape (num_segments, output_dim)
        """
        query = self.query.repeat(input.size(0), 1, 1)
        x, _ = self.attention(query=query, key=input, value=input)

        x = self.ffn(x)

        x = self.norm(x)

        return x


class SegAdapter(nn.Module):
    """
    Adapter module from the output of AlphaClip from the segmentation model to the input of the LLava model
    """

    def __init__(
        self,
        input_segment_dim: int,
        llava_embedding_dim: int,
        hidden_dim: int = None,
        expand_factor: int = 2,
        num_linears: int = 25,
        num_heads: int = [1],
        num_queries: int = [10],
        dropout: float = 0.25,
        noise_level: float = 1e-4,
        mlp_adapter: bool = False,
    ):
        """
        Adapter module from the output of AlphaClip from the segmentation model to the input of the LLava model

        Args:
            input_segment_dim (int): Dimension of the input segment embeddings
            llava_embedding_dim (int): Dimension of the LLava model's token embeddings
            num_linears (int): Number of linear layers to use in the adapter module
            hidden_dim (int): Dimension of the hidden layers in the adapter module (if None, hidden_dim = llava_embedding_dim)
            expand_factor (int): Factor to expand the hidden dimension in the FFN
            dropout (float): Dropout rate to apply in the adapter module
            num_heads (int): Number of heads to use in the multi-head attention layer
            num_queries (int): Number of queries to use in the multi-head attention layer
            blocks (int): Number of blocks to use in the adapter module
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = llava_embedding_dim

        self.mean_train_masks = torch.load("models/train_embs_mean.pt")
        self.input_norm = nn.LayerNorm(input_segment_dim)
        
        self.mlp_adapter = mlp_adapter
                
        if not mlp_adapter:
            self.linears = nn.ModuleList(
                [nn.Linear(input_segment_dim, hidden_dim) for _ in range(num_linears)]
            )

            blocks = []
            skips = []
            input_dim = hidden_dim
            output_dim = hidden_dim
            for i in range(len(num_heads)):
                if i == len(num_heads) - 1:
                    output_dim = llava_embedding_dim
                blocks.append(
                    QueryBlock(
                        input_dim,
                        output_dim,
                        num_queries[i],
                        expand_factor,
                        num_heads[i],
                        dropout,
                    )
                )
                skips.append(nn.Linear(input_dim, llava_embedding_dim, bias=False))
            self.blocks = nn.ModuleList(blocks)
            self.skips = nn.ModuleList(skips)

            self.final_skip = nn.Linear(input_segment_dim, llava_embedding_dim, bias=False)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_segment_dim, hidden_dim*expand_factor),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim*expand_factor, llava_embedding_dim),
            )
        
        self.norm = nn.LayerNorm(llava_embedding_dim)

        self.mean_emb = nn.Parameter(torch.zeros(llava_embedding_dim))
        self.std_emb = nn.Parameter(torch.ones(llava_embedding_dim))
        
        self.noise_level = noise_level

    def forward(self, segment_embeddings: torch.Tensor):
        """
        Forward pass through the adapter module

        Args:
            segment_embeddings (torch.Tensor): Segment embeddings from the segmentation model with shape (num_segments, input_segment_dim)

        Returns:
            llava_input (torch.Tensor): Input tensor for the LLava model with shape (seq_length, llava_embedding_dim)
        """
        segment_embeddings = segment_embeddings - self.mean_train_masks.to(segment_embeddings.device).to(segment_embeddings.dtype)
        segment_embeddings = self.input_norm(segment_embeddings)

        if not self.mlp_adapter:
            llava_input = self.final_skip(segment_embeddings)

            #print std and mean of the embeddings
            # print(f"Mean: {segment_embeddings.mean()} - Std: {segment_embeddings.std()}")
            x = [
                    linear(segment_embeddings + torch.randn_like(segment_embeddings) * self.noise_level * (1 if self.training else 0))
                    for linear in self.linears
                ]
            if len(x) == 0:
                x = segment_embeddings
                
            x = torch.stack(x, dim=1)
                
            for i in range(len(self.blocks)):
                llava_input += self.skips[i](x.mean(dim=1))
                x = self.blocks[i](x)
                
            x = x.mean(dim=1)
            llava_input += x
        else:
            segment_embeddings = segment_embeddings + torch.randn_like(segment_embeddings) * self.noise_level * (1 if self.training else 0)
            llava_input = self.mlp(segment_embeddings)

        llava_input = self.norm(llava_input)

        # normalize the output
        llava_input = llava_input * self.std_emb + self.mean_emb

        return llava_input


# Define the custom model class
class DynamicVocabLlavaModel(nn.Module):
    def __init__(self, model: PreTrainedModel, processor: LlavaProcessor):
        """
        Custom model class that wraps a pre-trained LLava model and adds functionality to add new tokens to the vocabulary
        and reset the token embeddings to the original state

        Args:
            model (PreTrainedModel): Pre-trained LLava model
            processor (LlavaProcessor): Processor object for the LLava model
            adapter (SegAdapter): Adapter module from the output of AlphaClip from the segmentation model to the input of the LLava model
        """
        super().__init__()
        self.llava_model = model
        self.original_vocab_size = model.config.text_config.vocab_size
        self.original_emb_matrix = model.get_input_embeddings().weight.clone().detach()
        self.tokenizer_vocab_size = processor.tokenizer.vocab_size
        self.processor = processor

    def forward(
        self,
        additional_tokens: torch.Tensor,
        num_generate: int = 1,
        reset_tokens: bool = False,
        token_masks: torch.Tensor = None,
        **kwargs,
    ):
        """
        Forward pass through the model

        Args:
            input_ids (torch.Tensor): Input tensor containing token IDs with shape (batch_size, seq_length)
            additional_tokens (torch.Tensor): Additional tokens to add to the vocabulary and the model's embedding layer with len batch_size and shape (num_tokens, seg_embedding_dim)
            num_generate (int): Number of tokens to generate
            reset_tokens (bool): Whether to reset the token embeddings to the original state after generating tokens (for gradients during training)
            **kwargs: Additional keyword arguments

        Returns:
            logits (torch.Tensor): Logits for the next tokens computed for the last num_generate tokens in the input sequence
        """
        # Add new tokens to the vocabulary and the model's embedding layer
        self.add_tokens(additional_tokens)

        # Prepare inputs for generation
        inputs = self.llava_model.prepare_inputs_for_generation(
            **kwargs, cache_position=torch.tensor([0])
        )
        inputs["num_logits_to_keep"] = num_generate
        # Forward pass through the model
        outputs = self.llava_model(
            **inputs,
            return_dict=True,
        )
        logits = outputs.logits

        if token_masks is not None:
            # logits is (batch_size, seq_length, vocab_size) and token_masks is (batch_size, vocab_size) so we need to expand token_masks
            token_masks = token_masks.unsqueeze(1).expand(-1, logits.size(1), -1)
            # Apply the token masks to the logits
            logits = logits * token_masks

        if DEBUG_PRINTS:
            # Generate next tokens
            next_tokens = torch.argmax(logits, dim=-1)
            generated = self.processor.tokenizer.decode(next_tokens[0])
            # print(f"Tokens: {next_tokens[0]}")
            print()
            print(f"GENERATED:\n\t{generated}")
            print()

        # Reset the token embeddings to the original state
        if reset_tokens:
            self.reset_tokens()

        return logits

    def add_tokens(self, new_tokens):
        if DEBUG_PRINTS:
            print(
                f"Adding {new_tokens.size(0)} new tokens to the vocabulary of size {self.original_vocab_size}"
            )
        # Get the current embedding layer
        embedding_layer = self.llava_model.get_input_embeddings()
        emb_weights = embedding_layer.weight.clone().detach()
        # Get the current number of tokens and embedding dimension
        old_num_tokens, embedding_dim = embedding_layer.weight.size()
        num_new_tokens = new_tokens.size(0)

        # Create a new embedding matrix with the new size
        new_embedding_weights = (
            torch.cat(
                [
                    emb_weights[: self.tokenizer_vocab_size + 1],
                    new_tokens,
                    emb_weights[self.tokenizer_vocab_size + 1 :],
                ]
            )
            .to(embedding_layer.weight.device)
            .to(embedding_layer.weight.dtype)
        )

        # Resize token embeddings
        self.llava_model.resize_token_embeddings(old_num_tokens + num_new_tokens)

        new_embedding_layer = self.llava_model.get_input_embeddings()
        # Assign the new embedding matrix to the embedding layer
        new_embedding_layer.weight = nn.Parameter(new_embedding_weights)

        # Update the LM head to have the same weights as the embeddings
        output_embedding_layer = self.llava_model.get_output_embeddings()
        output_embedding_layer.weight = new_embedding_layer.weight

    def reset_tokens(self):
        if DEBUG_PRINTS:
            print("Resetting the token embeddings")
        # Resize token embeddings
        self.llava_model.resize_token_embeddings(self.original_vocab_size)
        # Assign the original embedding matrix to the embedding layer
        embedding_layer = self.llava_model.get_input_embeddings()
        embedding_layer.weight = nn.Parameter(
            self.original_emb_matrix.to(embedding_layer.weight.device)
        )
        # Update the LM head to have the same weights as the embeddings
        output_embedding_layer = self.llava_model.get_output_embeddings()
        output_embedding_layer.weight = embedding_layer.weight


class LISA_Model(nn.Module):
    """
    Lisa model to be used during training
    """

    def __init__(
        self,
        model_name: str,
        seg_emb_size: int,
        lora_rank: int = 16,
        temperature: float = 0.1,
        mask_prob: float = 0.1,
        end_turn_token: str = "<end_of_turn>\n",
        seg_pos: str = "before",
        text: bool = True,
        pos_weight: int = 2,
        neg_weight: int = 1,
        q4: bool = True,
        q8: bool = False,
        dropout: float = 0.1,
        device: str = "cuda",
        **adapter_kwargs,
    ):
        """Initialize the LISA model

        Args:
            model_name (str): name of the llava model to load from huggingface
            seg_emb_size (int): size of the segment embeddings coming from the segmentation encoder model
            lora_rank (int, optional): Rank of the LoRA model. Defaults to 16.
            temperature (float, optional): Temperature to apply in the training. Defaults to 0.1.
            mask_prob (float, optional): Probability of masking a token in the input text. Defaults to 0.1.
            end_turn_token (str, optional): Token to be added at the end of the generated text. Defaults to "<end_of_turn>\n".
            seg_pos (str, optional): Position of the segment tokens in the generated text. Defaults to "before".
            text (bool, optional): Whether to use text as input to the model. Defaults to True.
            pos_weight (int, optional): Weight to assign to the positive mask tokens. Defaults to 2.
            neg_weight (int, optional): Weight to assign to the negative mask tokens. Defaults to 1.
            q4 (bool, optional): Load the model in 4-bit quantization. Defaults to True.
            q8 (bool, optional): Load the model in 8-bit quantization. Defaults to False.
            dropout (float, optional): Dropout rate to apply in the adapter module. Defaults to 0.1.
            device (str, optional): Device to run the model on. Defaults to "cuda"

        """
        super(LISA_Model, self).__init__()
        self.seg_emb_size = seg_emb_size
        self.seg_pos = seg_pos
        self.text = text
        self.mask_tok_prob = mask_prob

        # Initialize the processor
        processor = LlavaProcessor.from_pretrained(model_name)
        new_tokens = [f"<SEG_MASK_{i}>" for i in range(1, 1000)]
        processor.tokenizer.add_tokens(new_tokens)

        # Configuration for quantization
        assert not (q4 and q8), "Only one of q4 or q8 should be True"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=q4,
            load_in_8bit=q8,
        )

        # Load the pre-trained LLava model and wrap it with the custom model
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name, quantization_config=bnb_config
        )
        for param in model.parameters():
            param.requires_grad = False

        # Apply LoRA to the LLava model
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank*2,
            target_modules=[
                "q_proj",
                "v_proj",
            ],  # Adjust based on the actual module names
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        self.adapter = SegAdapter(
            seg_emb_size, model.get_input_embeddings().weight.size(1), dropout=dropout, **adapter_kwargs
        )

        self.llava_model = DynamicVocabLlavaModel(model, processor)
        self.temperature = temperature

        self.end_token = end_turn_token
        self.tokenized_end_token = processor.tokenizer.encode(
            end_turn_token, add_special_tokens=False
        )[0]

        self.device = device
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        
        self.to(device)

    def optim_step(
        self,
        texts: list[str],
        images: list[Image.Image],
        labels: list[str],
        pos_mask_embeds: list[torch.Tensor],
        neg_mask_embeds: list[torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ):
        """
        Forward pass + optimization step through the model

        Args:
            texts (list[str]): List of input text sequences I only the text
            images (list[Image.Image]): List of input images
            labels (list[str]): List of target text sequences for the model to predict I expect only the text that the model should predict
            pos_mask_embeds (list[torch.Tensor]): List of positive mask embeddings with shape (num_pos_masks, seg_emb_size)
            neg_mask_embeds (list[torch.Tensor]): List of negative mask embeddings with shape (num_neg_masks, seg_emb_size)
            optimizer (torch.optim.Optimizer): Optimizer object to update the model's parameters

        Returns:
            Tuple[torch.Tensor]: Logits for the next tokens computed for the last num_generate tokens in the input sequence and the loss
        """
        input_texts = []
        free_token = 1
        new_tokens = []

        num_new_tokens = sum(
            [pos_mask_embeds[i].size(0) for i in range(len(pos_mask_embeds))]
            + [neg_mask_embeds[i].size(0) for i in range(len(neg_mask_embeds))]
        )
        num_pos_tokens = sum([pos_mask_embeds[i].size(0) for i in range(len(pos_mask_embeds))])
        num_neg_tokens = sum([neg_mask_embeds[i].size(0) for i in range(len(neg_mask_embeds))])

        token_masks = torch.zeros(
            len(texts), self.llava_model.original_vocab_size + num_new_tokens
        ).to(self.device)
        token_masks[:, : self.llava_model.tokenizer_vocab_size + 1] = 1
        token_masks[:, self.llava_model.tokenizer_vocab_size + num_new_tokens + 1 :] = 1

        # Pass all tokens to the adapter and add the corresponding token lemma to the labels texts
        for i in range(len(pos_mask_embeds)):
            if self.seg_pos == "after" and self.text:
                labels[i] = labels[i] + " The segmentation mask for the object in the image "+ ("is " if pos_mask_embeds[i].size(0) > 1 else "are ")
            elif self.seg_pos == "before" and self.text:
                labels[i] = ". " + labels[i]
            for j in range(pos_mask_embeds[i].size(0)):
                new_tokens.append(pos_mask_embeds[i][j])
                if self.seg_pos == "before":
                    labels[i] = f" <SEG_MASK_{free_token}>{labels[i]}"
                elif self.seg_pos == "after":
                    labels[i] = (
                        f"{labels[i]}<SEG_MASK_{free_token}> "
                        if j < pos_mask_embeds[i].size(0) - 1
                        else f"{labels[i]}and <SEG_MASK_{free_token}>."
                    )
                token_masks[i, self.llava_model.tokenizer_vocab_size + free_token] = 1
                free_token += 1

            if self.seg_pos == "before" and self.text:
                labels[i] = "The segmentation mask for the object in the image "+ ("is" if pos_mask_embeds[i].size(0) > 1 else "are") + labels[i]

            # apply the chat template to the texts
            input_texts.append(
                self.llava_model.processor.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": f"<image>\n{texts[i]} Output the segmentation mask for the object in the image"},
                        {"role": "assistant", "content": labels[i]},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            # get the last token of the text as the end token to be added to the labels
            labels[i] += f"{self.end_token}"

        if DEBUG_PRINTS:
            print()
            print("MODEL INPUTS")
            print(f"\tInput texts: {input_texts[0]}")
            print(f"\tLabels: {labels[0]}")
            print()

        for i in range(len(neg_mask_embeds)):
            for j in range(neg_mask_embeds[i].size(0)):
                new_tokens.append(neg_mask_embeds[i][j])
                token_masks[i, self.llava_model.tokenizer_vocab_size + free_token] = 1
                free_token += 1

        new_tokens = torch.stack(new_tokens)
        new_tokens = self.adapter(new_tokens.to(self.device))

        # tokenize the texts
        inputs = self.llava_model.processor(
            text=input_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(self.device)
        # remove last token from the input text
        inputs["input_ids"] = inputs["input_ids"][:, :-1]
        # drop random tokens and substitute them with the unk token
        mask = torch.rand(inputs["input_ids"].size()) < self.mask_tok_prob
        inputs["input_ids"][mask] = self.llava_model.processor.tokenizer.unk_token_id

        labels_ids = self.llava_model.processor(
            text=labels,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        )
        labels_input_ids = labels_ids["input_ids"].to(self.device)

        # print()
        # print("MODEL TOKENS INPUT")
        # print(f"\tInput tokens: {inputs['input_ids']}")
        # print(f"\tLabels: {labels_input_ids}")
        # print()

        # Forward pass through the model
        logits = self.llava_model(
            **inputs,
            additional_tokens=new_tokens,
            num_generate=labels_input_ids.size(1),
            reset_tokens=False,
            token_masks=token_masks,
        )

        logits = logits.view(-1, logits.size(-1)) / self.temperature

        labels_input_ids = labels_input_ids.reshape(-1)

        class_weights = torch.ones(logits.size(-1)).to(self.device)
        class_weights[
            self.llava_model.tokenizer_vocab_size + 1 :
            self.llava_model.tokenizer_vocab_size + num_pos_tokens + 1
        ] = self.pos_weight
        class_weights[
            self.llava_model.tokenizer_vocab_size + num_pos_tokens + 1 :
            self.llava_model.tokenizer_vocab_size + num_new_tokens + 1
        ] = self.neg_weight
        loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.llava_model.processor.tokenizer.pad_token_id,
            reduction="none",
            weight=class_weights,
        )
        # where the labels_input_ids are one of the new tokens, the loss should be multiplied by 2
        loss = loss_fn(logits, labels_input_ids)

        weights = torch.ones_like(labels_input_ids)
        # weights[labels_input_ids > self.llava_model.tokenizer_vocab_size] = 1.5
        # # penilize the model for not generating the end tokens
        # weights[labels_input_ids == self.tokenized_end_token] = 1.5
        loss = (loss * weights).mean()
        
        # if loss is nan break
        if torch.isnan(loss):
            print("NAN LOSS")
            return None, None

        # backward pass
        optimizer.zero_grad()

        loss.backward()

        # copy the gradients to the mask embeddings
        emb_grads = self.llava_model.llava_model.get_input_embeddings().weight.grad
        new_tokens.backward(
            emb_grads[
                self.llava_model.tokenizer_vocab_size + 1 : self.llava_model.tokenizer_vocab_size
                + num_new_tokens
                + 1
            ]
        )

        optimizer.step()

        self.llava_model.reset_tokens()

        return logits, loss

    def generate(
        self,
        texts: list[str],
        images: list[Image.Image],
        pos_mask_embeds: list[torch.Tensor],
        neg_mask_embeds: list[torch.Tensor],
        max_new_tokens: int = 100,
        n_beams: int = 1,
        repetition_penalty: float = 1.0,
        do_sample: bool = False,
        temperature: float = 0.1,
    ):
        """
        Generate text from the model

        Args:
            texts (list[str]): List of input text sequences
            images (list[Image.Image]): List of input images
            pos_mask_embeds (list[torch.Tensor]): List of positive mask embeddings with shape (num_pos_masks, seg_emb_size)
            neg_mask_embeds (list[torch.Tensor]): List of negative mask embeddings with shape (num_neg_masks, seg_emb_size)
            max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 100.
            n_beams (int, optional): Number of beams to use in the generation. Defaults to 1.
            repetition_penalty (float, optional): Repetition penalty to apply in the generation. Defaults to 1.0.
            do_sample (bool, optional): Whether to sample or not during generation. Defaults to False.
            temperature (float, optional): Temperature to apply in the generation. Defaults to 0.1.

        Returns:
            Tuple[list[str], list[torch.Tensor]]: List of generated text sequences and the corresponding token IDs
        """
        outputs = []
        tokens = []
        for i in range(len(texts)):
            new_tokens = torch.cat([pos_mask_embeds[i], neg_mask_embeds[i]])
            new_tokens = self.adapter(new_tokens.to(self.device))
            self.llava_model.add_tokens(new_tokens)

            # apply the chat template to the texts
            input_text = self.llava_model.processor.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": f"<image>\n{texts[i]}"},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            
            # tokenize the texts
            inputs = self.llava_model.processor(
                text=input_text,
                images=images[i],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(self.device)

            # call generate on the model
            generated_tok = self.llava_model.llava_model.generate(
                **inputs,
                num_beams=n_beams,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenized_end_token,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
            )

            generated = self.llava_model.processor.batch_decode(
                generated_tok,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )[0]
            # get only from model\n afterwords
            generated = generated.split("model\n")[1]
            self.llava_model.reset_tokens()

            outputs.append(generated)
            tokens.append(generated_tok[0])
        return outputs, tokens

    def forward(self, **kwargs):
        if self.training:
            return self.optim_step(**kwargs)
        else:
            return self.generate(**kwargs)
