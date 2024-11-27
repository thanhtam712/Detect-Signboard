import sys
import cv2
import json
import zipfile
import argparse

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from numpy import ndarray
from typing import Dict, List, Union, Any

sys.path.append(str(Path(__file__).parent.parent.parent / "models" / "D-FINE"))
import torch
import torch.nn as nn
from src.core import YAMLConfig  # type: ignore
import torchvision.transforms as T

sys.path.append(str(Path(__file__).parent.parent))
from utils import convert_xyxy_to_yolo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="dataset/data_test/images/", help="Image file")
    parser.add_argument("--checkpoint", help="Checkpoint file")
    parser.add_argument("--visualize-out", type=str, default="vis_dfine", help="Path to output file")
    parser.add_argument("--sub-folder", default="dfine", help="Path to output file")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--score-thr", type=float, default=0.2)

    parser.add_argument("--config", help="Config file")
    parser.add_argument("--img-size", type=int, default=640)
    
    args = parser.parse_args()
    return args



def infer_dfine(args, images: Dict[str, ndarray]):
    """
    Inference using DFine model

    Args:
        args: parsed arguments
        images: dictionary of images, where key is image name and value is image array

    Returns:
        list of results
    """

    # init model
    cfg = YAMLConfig(args.config, resume=args.checkpoint)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        raise AttributeError("Only support checkpoint to load model.state_dict by now.")

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(
            self,
        ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to("cuda")

    img_paths = list(images.keys())
    imgs = [images[img_path] for img_path in img_paths]
    results = []

    for id_image, img in tqdm(list(enumerate(imgs))):
        img = Image.fromarray(img).convert("RGB")
        width, height = img.size
        orig_size = torch.tensor([width, height])[None].to("cuda:0")

        transforms = T.Compose(
            [
                T.Resize((args.img_size, args.img_size)),
                T.ToTensor(),
            ]
        )
        im_data = transforms(img)[None].to("cuda:0")  # type: ignore
        output = model(im_data, orig_size)

        with torch.no_grad():
            labels, boxes, scores = output
            labels = labels[0].detach().cpu().numpy()
            boxes = boxes[0].detach().cpu().numpy()
            scores = scores[0].detach().cpu().numpy()

            for i in range(len(labels)):
                if scores[i] < args.score_thr or labels[i] not in [0, 1, 2, 3]:
                    continue

                x_center, y_center, w, h = convert_xyxy_to_yolo(
                    boxes[i], (width, height)
                )

                results.append(
                    {
                        "image_id": img_paths[id_image],
                        "category_id": int(labels[i]),
                        "bbox": [x_center, y_center, w, h],
                        "score": scores[i],
                        "width": width,
                        "height": height,
                    }
                )

    return results


def visualize_results(args, results: List[dict]):
    """
    Visualize results

    Args:
        args: parsed arguments
        results: list of results, where each result is a dictionary with keys:
            - image_id: image name
            - category_id: category id
            - bbox: bounding box in YOLO format
            - score: confidence score

    Returns:
        None
    """
    path_visualize_out = Path("./source/visualize") / args.visualize_out
    (path_visualize_out).mkdir(exist_ok=True, parents=True)

    colors = {
        0: (255, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
        3: (255, 255, 0),
    }

    for result in tqdm(results):
        image_id = result["image_id"]
        if result["score"] < args.score_thr:
            continue
        
        file_visualize = path_visualize_out / image_id
        if file_visualize.exists():
            img = cv2.imread(file_visualize)
        else:
            img = cv2.imread(Path(args.folder) / result["image_id"])

        assert img is not None, f"Image not found: {result['image_id']}"

        x_center, y_center, w, h = result["bbox"]
        x1 = int((x_center - w / 2) * img.shape[1])
        y1 = int((y_center - h / 2) * img.shape[0])
        x2 = int((x_center + w / 2) * img.shape[1])
        y2 = int((y_center + h / 2) * img.shape[0])

        cv2.rectangle(img, (x1, y1), (x2, y2), colors[result["category_id"]], 2)
        cv2.putText(
            img,
            f"{result['category_id']}_{result['score']:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colors[result["category_id"]],
            2,
        )

        cv2.imwrite(file_visualize, img)

def save_submission(sub_folder: Path, results: List[Dict[str, Any]]):
    """
    Save the results in the submission format.

    Args:
        out_file_path (str): Path to the output file.
        results (List[Dict[str, Any]]): List of results. Each result is a dictionary with the following keys:
            - image_id (str): Image ID.
            - category_id (int): Category ID.
            - bbox (List[float]): Bounding box in YOLO format.
            - score (float): Confidence score.
    """
    path_folder = Path("./submissions") / sub_folder
    path_folder.mkdir(exist_ok=True)

    out_file = path_folder / "answer.txt"
    
    with open(out_file, "a+") as file_txt:
        results_bbox_scores = []
        for result in tqdm(results, desc="Saving submission"):
            image_id = result["image_id"]
            category_id = 0
            bbox = result["bbox"]
            score = result["score"]

            bbox = [round(x, 5) for x in bbox]
            # score = round(score, 5)
            file_txt.write(
                f"{Path(image_id).stem} {int(category_id)} {' '.join(map(str, bbox))}\n"
            )
            x_min = bbox[0] - bbox[2] / 2
            y_min = bbox[1] - bbox[3] / 2
            x_max = bbox[0] + bbox[2] / 2
            y_max = bbox[1] + bbox[3] / 2
            
            if x_min < 0:
                x_min = 0
            if y_min < 0:
                y_min = 0
            if x_max > 1:
                x_max = 1
            if y_max > 1:
                y_max = 1
            
            results_bbox_scores.append({"image_name": Path(image_id).stem, "bbox": [x_min, y_min, x_max, y_max], "score": float(score)})
        
    with zipfile.ZipFile(path_folder / (str(sub_folder) + ".zip") , "w") as f:
        f.write(path_folder / "answer.txt", "answer.txt")

    with open (path_folder / "results.json", "w") as f:
        json.dump(results_bbox_scores, f, indent=4)

def main():
    args = parse_args()

    images = {}
    l_images = list(Path(args.folder).iterdir())

    print("Start loading images...")
    for img_path in tqdm(l_images):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images[Path(img_path).name] = img
        
    results = infer_dfine(args, images)
    
    # save visualization
    if args.visualize_out:
        print("Visualizing results...")
        visualize_results(args, results)

    # save results
    if args.sub_folder:
        print("Saving results...")
        save_submission(Path(args.sub_folder), results)


if __name__ == "__main__":
    
    sys.path.insert(0, 
                    str(Path(__file__).parent.parent) if sys.version_info[1] >=9 else
                    str(Path(__file__).absolute().parent.parent)
                )
    main()
    
    
""" CUDA_VISIBLE_DEVICES=4 python source/tools/inference_dfine.py --folder dataset/data_test/images/ --checkpoint models/D-FINE/output/dfine_hgnetv2_x_coco/last.pth --sub-folder dfine --batch-size 4 --config models/D-FINE/configs/dfine/dfine_hgnetv2_x_coco.yml --img-size 640 --visualize-out dfine"""
