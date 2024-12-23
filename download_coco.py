import fiftyone.zoo as foz
import fiftyone as fo
from configuration import load_yaml_config

conf = load_yaml_config("config.yaml")

fo.config.dataset_zoo_dir = conf.dataset.COCO.dataset_zoo_dir

# deterministic shuffling

dataset = foz.load_zoo_dataset(
    conf.dataset.COCO.name,
    split="train",
    label_types=["segmentations"],
    max_samples=conf.dataset.COCO.num_samples,
    shuffle=True,
    seed=42,
)
