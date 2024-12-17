import json
import logging
import os

import matplotlib.pyplot as plt
import torch
from PIL import Image

from configuration import ProjectConfig, dataclass, load_yaml_config
from llava_finetune.functions import load_model
from llava_finetune.model import LISA_Model
from llava_finetune.utils import draw_shapes
from preprocess import PreprocessPipeline


@dataclass
class InferenceSample:
    query: str
    image: str
    new_tokens: list[int] = None
    new_tokens_shapes: list[list[int]] = None


class InferencePipeline:
    """
    Inference Pipeline for the LISA ACV project
    """

    logger = logging.getLogger(__name__)

    def __init__(self, config: ProjectConfig, model_name="longer"):
        self.config = config
        self.logger.info("Loading PreprocessPipeline")

        default_additional_preprocess_params = {
            "model": "alpha-clip",
            "only_masks": False,
        }
        # search for preprocess_model_name.json and overwrite the default_additional_preprocess_params
        preprocess_model_name = f"preprocess_{model_name}.json"
        preprocess_model_name_path = os.path.join("preprocess", preprocess_model_name)
        if os.path.exists(preprocess_model_name_path):
            with open(preprocess_model_name_path, "r") as f:
                additional_preprocess_params = json.load(f)
            default_additional_preprocess_params.update(additional_preprocess_params)

        self.only_masks = default_additional_preprocess_params.get("only_masks", False)
        # add them to the config
        config.additional_preprocess_params = default_additional_preprocess_params

        self.pp = PreprocessPipeline(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.logger.info("Loading model")
        self.model: LISA_Model = load_model(
            f"models/{model_name}.pth", f"models/{model_name}.json", self.device
        ).eval()

        self.tokenizer = self.model.llava_model.processor.tokenizer

    def token_similarity(
        self,
        data: list[InferenceSample | dict],
        softmax: bool = True,
        temperature: float = 1.0,
        cosine_similarity: bool = False,
    ) -> list[dict]:
        """
        Calculate the token similarity for a given image of each one of the produced semantic masks embeddings.

        Args:
            data: list of InferenceSamples
            softmax: whether to use softmax or not
            temperature: temperature to use in softmax
            cosine_similarity: whether to use cosine similarity or not

        Returns:
            list of dictionaries containing the token similarities for each image and the masks shapes ("token_similarities", "embs_similarities", "masks")
        """

        for d in data:
            if isinstance(d, dict):
                d = InferenceSample(**d)

        self.logger.info("Calculating token similarity")

        if len(data) == 0:
            return []

        with torch.no_grad():
            preprocessed = [self.pp.inference_preprocess(d.image, self.only_masks) for d in data]    
            for i, d in enumerate(data):
                if d.new_tokens is not None:
                    preprocessed[i]["sam_embs"] += d.new_tokens
                    preprocessed[i]["sam_shapes"] += d.new_tokens_shapes

            embs = [
                self.model.adapter(torch.tensor(res.get("sam_embs")).to(self.device))
                for res in preprocessed
            ]
            token_mat = self.model.llava_model.original_emb_matrix.T.to(embs[0].dtype)

            # normalize embeddings if cosine similarity
            if cosine_similarity:
                embs = [emb / emb.norm(dim=1, keepdim=True) for emb in embs]
                token_mat = token_mat / token_mat.norm(dim=0, keepdim=True)

            token_similarities = [torch.matmul(emb, token_mat) / temperature for emb in embs]
            embs_similarities = [torch.matmul(emb, emb.T) / temperature for emb in embs]

            if softmax:
                token_similarities = [torch.softmax(sim, dim=0) for sim in token_similarities]
                embs_similarities = [torch.softmax(sim, dim=0) for sim in embs_similarities]

            masks = [res.get("sam_shapes") for res in preprocessed]

            for i in range(len(data)):
                yield {
                    "token_similarities": token_similarities[i],
                    "embs_similarities": embs_similarities[i],
                    "masks": masks[i],
                }

    def inference(
        self,
        data: list[InferenceSample | dict],
        max_new_tokens: int = 100,
        n_beams: int = 1,
        repeat_penalty: float = 2.0,
        temperature: float = 0.8,
        do_sample: bool = False,
    ) -> list[dict]:
        """
        Perform inference on a list of InferenceSamples


        Args:
            data: list of InferenceSamples
            max_new_tokens: maximum number of tokens to generate
            n_beams: number of beams to use in beam search
            repeat_penalty: repetition penalty to use in beam search
            temperature: temperature to use in beam search
            do_sample: whether to sample or not during generation

        Returns:
            list of dictionaries containing the generated text, masks and chosen tokens
        """
        for d in data:
            if isinstance(d, dict):
                d = InferenceSample(**d)

        self.logger.info("Performing inference with n_beams: ", n_beams)
        with torch.no_grad():
            queries = [d.query for d in data]
            img_paths = [d.image for d in data]

            preprocessed = [self.pp.inference_preprocess(img, self.only_masks) for img in img_paths]

            embs = [res.get("sam_embs") for res in preprocessed]
            masks = [res.get("sam_shapes") for res in preprocessed]
            for i, d in enumerate(data):
                if d.new_tokens is not None:
                    embs[i] += d.new_tokens
                    masks[i] += d.new_tokens_shapes

            images = [Image.open(img_path).convert("RGBA") for img_path in img_paths]

            gen_texts, gen_tokens = self.model.generate(
                queries,
                images,
                [torch.tensor([]) for q in queries],
                [torch.tensor(emb) for emb in embs],
                max_new_tokens=max_new_tokens,
                n_beams=n_beams,
                repetition_penalty=float(repeat_penalty),
                do_sample=do_sample,
                temperature=temperature,
            )

            chosen_tokens = [
                list(
                    set(
                        (
                            gen_tokens[i][
                                gen_tokens[i] > self.model.llava_model.tokenizer_vocab_size
                            ]
                            - self.model.llava_model.tokenizer_vocab_size
                            - 1
                        ).tolist()
                    )
                )
                for i in range(len(gen_tokens))
            ]

            for i in range(len(data)):
                yield {
                    "gen_text": gen_texts[i],
                    "masks": [x for j, x in enumerate(masks[i]) if j in chosen_tokens[i]],
                    "chosen_tokens": chosen_tokens[i],
                }


if __name__ == "__main__":
    config = load_yaml_config("config.yaml")

    ip = InferencePipeline(config, "shorter_big")

    data = [
        InferenceSample(
            query="Which vehicle should I sleep in?",
            image="inference/2593366765_589ca5148e_o.jpg",
        ),
        InferenceSample(
            query="Where is the van?",
            image="inference/2593366765_589ca5148e_o.jpg",
        ),
        InferenceSample(
            query="Where is the roulotte?",
            image="inference/2593366765_589ca5148e_o.jpg",
        ),
        InferenceSample(
            query="Is there a ladder in this image?",
            image="inference/2593366765_589ca5148e_o.jpg",
        ),
        InferenceSample(
            query="Are there hot singles in the image?",
            image="inference/2593366765_589ca5148e_o.jpg",
        ),
        InferenceSample(
            query="Is there a vehicle with two wheels in the image?",
            image="inference/2593366765_589ca5148e_o.jpg",
        ),
    ]

    res = ip.inference(data, n_beams=5)

    for i, r in enumerate(res):
        print("Query: ", data[i].query)
        print("Answer: ", r["gen_text"])

        comb = draw_shapes(Image.open(data[i].image), r["masks"])

        plt.figure(figsize=(12, 8))
        plt.imshow(comb)
        plt.axis("off")
        plt.savefig(f"output/inference/inf_{i}.png")
