import numpy as np
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
import wandb
import json
import argparse
from inference import InferencePipeline, InferenceSample
from llava_finetune.utils import get_dataloaders
from configuration import load_yaml_config
import torch
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

# seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

def arg_parser():
    parser = argparse.ArgumentParser(description="Validation")
    # parser.add_argument("--model_name", type=str, default="MLP-adapter-big-outputLora", help="Name of the model to be used for validation")
    parser.add_argument("--train_json", type=str, default="data/train-v3.jsonl", help="Path to the training JSON file")
    parser.add_argument("--val_json", type=str, default="data/val-v3.jsonl", help="Path to the validation JSON file")
    parser.add_argument("--test_json", type=str, default="data/test-v3.jsonl", help="Path to the test JSON file")
    return parser.parse_args()

def shapes_to_mask(shapes, size=(1024, 1024)):
    """Convert list of shapes to binary mask"""
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    
    for shape in shapes:
        for polygon in shape:
            points = [tuple(point) for point in polygon]
            if len(points) >= 2:
                draw.polygon(points, fill=1)
    
    return np.array(mask, dtype=bool)

def compute_IoU(gt_shapes, pred_shapes):
    """
    Compute IoU and gIoU using Shapely geometry operations.

    Args:
        gt_shapes (list): List of ground truth polygons, each polygon is a list of [x,y] points.
        pred_shapes (list): List of predicted polygons, each polygon is a list of [x,y] points.

    Returns:
        dict: Dictionary containing IoU and gIoU scores as well as intersection and union areas.
    """
    # Convert shapes into valid Shapely Polygons
    # 1,2,48
    gt_polygons = []
    for shape in gt_shapes:
        for poly in shape:
            if len(poly) >= 3:
                p = Polygon(poly)
                if p.is_valid and not p.is_empty:
                    gt_polygons.append(p)
                else:
                    tqdm.write("Invalid GT polygon")
            else:
                tqdm.write("Invalid GT polygon not enough points")
    
    # 2,1,57
    pred_polygons = []
    for shape in pred_shapes:
        for poly in shape:
            if len(poly) >= 3:
                p = Polygon(poly)
                if p.is_valid and not p.is_empty:
                    pred_polygons.append(p)
                else:
                    tqdm.write("Invalid prediction polygon")
            else:
                tqdm.write("Invalid prediction polygon not enough points")
    
    # If no valid polygons in either GT or prediction, IoU and gIoU are zero
    if len(gt_polygons) == 0 or len(pred_polygons) == 0:
        tqdm.write("No valid polygons in either GT or prediction")
        return {
            "intersection": 0.0,
            "union": 0.0,
            "IoU": 0.0,
            "gIoU": 0.0
        }
    
    # Create union polygons
    gt_union = unary_union(gt_polygons)
    pred_union = unary_union(pred_polygons)
    
    # Intersection and union areas
    intersection_area = gt_union.intersection(pred_union).area
    union_area = gt_union.union(pred_union).area
    
    IoU = intersection_area / union_area if union_area > 0 else 0.0
    
    # If no intersection or union is zero, IoU = 0 and so gIoU = 0
    if union_area == 0:
        tqdm.write("Union area is zero")
        return {
            "intersection": intersection_area,
            "union": union_area,
            "IoU": IoU,
            "gIoU": 0.0
        }

    # Compute the convex hull of the combined geometry
    combined_geometry = gt_union.union(pred_union)
    convex_hull = combined_geometry.convex_hull
    convex_hull_area = convex_hull.area
    
    # gIoU = IoU - ((ConvexHullArea - UnionArea) / ConvexHullArea)
    # Ensure convex_hull_area > 0
    if convex_hull_area == 0:
        # Degenerate case: everything collapses to a line or point
        tqdm.write("Convex hull area is zero")
        gIoU = IoU
    else:
        gIoU = IoU - ((convex_hull_area - union_area) / convex_hull_area)
    
    return {
        "intersection": intersection_area,
        "union": union_area,
        "IoU": IoU,
        "gIoU": gIoU
    }


if __name__ == "__main__":
    args = arg_parser()

    config = load_yaml_config("config_davide.yaml")
    wan_db_token = config.others.wandb_token
    wandb.login(key=wan_db_token)

    # Initialize data loaders
    data_loader, data_val_loader, data_test_loader = get_dataloaders(
        config.dataset.json_path,
        config.dataset.image_dir,
        1,
        args.train_json,
        args.val_json,
        args.test_json
    )
    
    # Initialize inference pipeline
    models = ["LONG", "MLP-adapter-big-outputLora-randomized", "TEST_REDUCT_NEG"]
    for model_name in models:
        # get model json parameters
        model_params = "models/" + model_name + ".json"
        with open(model_params, "r") as f:
            model_params = json.load(f)
        # initialize the wandb project
        wandb.init(project="ACV_testing", name=f"{model_name}", config=model_params)

        inference_pipeline = InferencePipeline(config, model_name)
        output_seg_query = " Output the segmentation mask for the object in the image"

        # Validation phase
        val_intersection = 0
        val_intersection_with_gt_masks = 0
        val_union = 0
        val_union_with_gt_masks = 0
        val_iou_list = []
        val_iou_list_with_gt_masks = []
        val_giou_list = []
        val_giou_list_with_gt_masks = []

        for data in tqdm(data_val_loader):
            samples = [InferenceSample(data["queries"][i]+output_seg_query, data["image_path"][i]) for i in range(len(data["queries"]))]
            results = inference_pipeline.inference(samples, n_beams=1, temperature=0.1, repeat_penalty=1.0, max_new_tokens=200)
            
            samples_with_gt_masks = [InferenceSample(data["queries"][i]+output_seg_query, data["image_path"][i], data["gt_embs"][i], data["gt_shapes"][i]) for i in range(len(data["queries"]))]
            results_with_gt_masks = inference_pipeline.inference(samples_with_gt_masks, n_beams=1, temperature=0.1, repeat_penalty=1.0, max_new_tokens=200)

            i = 0
            for result, result_with_gt_masks in zip(results, results_with_gt_masks):
                gt_shapes = data["gt_shapes"][i]
                
                pred_shapes = result["masks"]
                pred_shapes_with_gt_masks = result_with_gt_masks["masks"]

                metrics = compute_IoU(gt_shapes, pred_shapes)
                metrics_with_gt_masks = compute_IoU(gt_shapes, pred_shapes_with_gt_masks)

                val_intersection += metrics["intersection"]
                val_union += metrics["union"]
                val_giou_list.append(metrics["gIoU"])
                val_iou_list.append(metrics["IoU"])
                
                val_intersection_with_gt_masks += metrics_with_gt_masks["intersection"]
                val_union_with_gt_masks += metrics_with_gt_masks["union"]
                val_giou_list_with_gt_masks.append(metrics_with_gt_masks["gIoU"])
                val_iou_list_with_gt_masks.append(metrics_with_gt_masks["IoU"])
                
                wandb.log({
                    "val/gIoU_per_sample": metrics["gIoU"],
                    "val/IoU_per_sample": metrics["IoU"],
                    "val/gIoU_per_sample_with_gt_masks": metrics_with_gt_masks["gIoU"],
                    "val/IoU_per_sample_with_gt_masks": metrics_with_gt_masks["IoU"]
                })
                i += 1

        val_giou = np.mean(val_giou_list)
        val_ciou = val_intersection / val_union if val_union > 0 else 0.0
        val_iou = np.mean(val_iou_list)
        
        val_giou_with_gt_masks = np.mean(val_giou_list_with_gt_masks)
        val_ciou_with_gt_masks = val_intersection_with_gt_masks / val_union_with_gt_masks if val_union_with_gt_masks > 0 else 0.0
        val_iou_with_gt_masks = np.mean(val_iou_list_with_gt_masks)

        wandb.log({
            "val/mean_gIoU": val_giou,
            "val/cIoU": val_ciou,
            "val/mean_gIoU_with_gt_masks": val_giou_with_gt_masks,
            "val/cIoU_with_gt_masks": val_ciou_with_gt_masks,
            "val/mean_IoU": val_iou,
            "val/mean_IoU_with_gt_masks": val_iou_with_gt_masks
        })

        print(f"Validation gIoU: {val_giou}")
        print(f"Validation cIoU: {val_ciou}")
        print(f"Validation gIoU with gt masks: {val_giou_with_gt_masks}")
        print(f"Validation cIoU with gt masks: {val_ciou_with_gt_masks}")

        # # Test phase
        # test_intersection = 0
        # test_union = 0
        # test_giou_list = []
        # test_iou_list = []
        
        # test_intersection_with_gt_masks = 0
        # test_union_with_gt_masks = 0
        # test_giou_list_with_gt_masks = []
        # test_iou_list_with_gt_masks = []

        # for data in tqdm(data_test_loader):
        #     samples = [InferenceSample(data["queries"][i]+output_seg_query, data["image_path"][i]) for i in range(len(data["queries"]))]
        #     results = inference_pipeline.inference(samples, n_beams=1, temperature=1.0, repeat_penalty=1.0)
            
        #     samples_with_gt_masks = [InferenceSample(data["queries"][i]+output_seg_query, data["image_path"][i], data["gt_embs"][i], data["gt_shapes"][i]) for i in range(len(data["queries"]))]
        #     results_with_gt_masks = inference_pipeline.inference(samples_with_gt_masks, n_beams=1, temperature=1.0, repeat_penalty=1.0)

        #     i = 0
        #     for result, result_with_gt_masks in zip(results, results_with_gt_masks):
        #         gt_shapes = data["gt_shapes"][i]
        #         pred_shapes = result["masks"]

        #         metrics = compute_IoU(gt_shapes, pred_shapes)
        #         metrics_with_gt_masks = compute_IoU(gt_shapes, pred_shapes_with_gt_masks)

        #         test_intersection += metrics["intersection"]
        #         test_union += metrics["union"]
        #         test_giou_list.append(metrics["gIoU"])
        #         test_iou_list.append(metrics["IoU"])
                
        #         test_intersection_with_gt_masks += metrics_with_gt_masks["intersection"]
        #         test_union_with_gt_masks += metrics_with_gt_masks["union"]
        #         test_giou_list_with_gt_masks.append(metrics_with_gt_masks["gIoU"])
        #         test_iou_list_with_gt_masks.append(metrics_with_gt_masks["IoU"])

        #         wandb.log({
        #             "test/gIoU_per_sample": metrics["gIoU"],
        #             "test/gIoU_per_sample_with_gt_masks": metrics_with_gt_masks["gIoU"],
        #             "test/IoU_per_sample": metrics["IoU"],
        #             "test/IoU_per_sample_with_gt_masks": metrics_with_gt_masks["IoU"]
        #         })
        #         i += 1

        # test_giou = np.mean(test_giou_list)
        # test_ciou = test_intersection / test_union if test_union > 0 else 0.0
        # test_iou = np.mean(test_iou_list)
        
        # test_giou_with_gt_masks = np.mean(test_giou_list_with_gt_masks)
        # test_ciou_with_gt_masks = test_intersection_with_gt_masks / test_union_with_gt_masks if test_union_with_gt_masks > 0 else 0.0
        # test_iou_with_gt_masks = np.mean(test_iou_list_with_gt_masks)

        # wandb.log({
        #     "test/mean_gIoU": test_giou,
        #     "test/cIoU": test_ciou,
        #     "test/mean_gIoU_with_gt_masks": test_giou_with_gt_masks,
        #     "test/cIoU_with_gt_masks": test_ciou_with_gt_masks,
        #     "test/mean_IoU": test_iou,
        #     "test/mean_IoU_with_gt_masks": test_iou_with_gt_masks
        # })

        # print(f"Test gIoU: {test_giou}")
        # print(f"Test cIoU: {test_ciou}")
        # print(f"Test gIoU with gt masks: {test_giou_with_gt_masks}")
        # print(f"Test cIoU with gt masks: {test_ciou_with_gt_masks}")

        wandb.finish()
        # empty GPU memory after each model
        del inference_pipeline
        torch.cuda.empty_cache()
