import os
import sys
import cv2
import math
import json
import zipfile
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path
from ensemble_boxes import *

sys.path.append(str(Path(__file__).parent.parent))
from utils.bbox import calculate_iou


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-json", nargs="+", help="Path to json file")
    parser.add_argument(
        "--sub-folder",
        type=str,
        required=True,
        default="fusion",
        help="Path to output file",
    )
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--skip-box-thr", type=float, default=0.4)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--weights", nargs="+", type=float, default=[1, 1])
    parser.add_argument("--method", type=str, default="weighted_boxes_fusion")
    parser.add_argument("--output", type=str, default="output.json")

    return parser.parse_args()


def visualize(image: str, list_boxes: list[float], list_scores: float, output: str):

    args = parse_args()

    if os.path.exists(output):
        print(f"Error: The file {output} already exists.")
        return

    if not os.path.exists(image):
        image = image.replace(".jpg", ".png")
        if not os.path.exists(image):
            print(f"Error: The file {image} does not exist.")
            return

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
    models = ["yolov8", "yolov8_augement", "dfine", "dfine_augement"]

    img = cv2.imread(image)
    if img is None:
        print(f"Error: Unable to read the file {image}. Check format and path.")
        return

    for idx, (boxes, scores) in enumerate(zip(list_boxes, list_scores)):
        for box, score in zip(boxes, scores):
            if score < args.skip_box_thr:
                continue

            x_min, y_min, x_max, y_max = box

            width, height = img.shape[1], img.shape[0]
            
            if math.isnan(x_min) or math.isnan(y_min) or math.isnan(x_max) or math.isnan(y_max) or math.isnan(score):
                    continue
            
            x_min, y_min, x_max, y_max = (
                int(x_min * width),
                int(y_min * height),
                int(x_max * width),
                int(y_max * height),
            )
            
            img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), colors[idx], 2)
            img = cv2.putText(
                img,
                f"{score:.2f} {models[idx]}",
                (x_min, y_min),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                colors[idx],
                2,
            )

    cv2.imwrite(output, img)


def check_missing_boxes(boxes: list[list[float]], scores: list[float], labels: list[int], boxes_list: list[list[list[float]]], scores_list: list[list[float]], labels_list: list[list[int]]):
    """
    Check if there are missing boxes in the final result
    
    Args:
        boxes: list of boxes
        scores: list of scores
        labels: list of labels
        boxes_list: list of boxes from all models
        scores_list: list of scores from all models
        labels_list: list of labels from all models
        
    Returns:
        Boxes, scores, labels after checking
    """
    boxes = boxes.tolist()
    scores = scores.tolist()
    labels = labels.tolist()
    
    for boxes_model, scores_model, labels_model in zip(boxes_list, scores_list, labels_list):
        for box, score, label in zip(boxes_model, scores_model, labels_model):
            for box_pred in boxes:
                if calculate_iou(box, box_pred) == 0 and score > 0.7:
                    boxes.append(box)
                    scores.append(score)
                    labels.append(label)
                    
    return boxes, scores, labels


def main():
    args = parse_args()

    sub_folder = Path("./submissions") / ("fusion_" + args.sub_folder)
    (sub_folder).mkdir(exist_ok=True)
    file_txt = sub_folder / "answer.txt"

    folder_visualize = Path(
        "/mlcv2/WorkingSpace/Personal/baotg/TTam/you_know/source/visualize"
    ) / ("fusion_" + args.sub_folder)
    (folder_visualize).mkdir(exist_ok=True)

    folder_visualize_multi_model = Path(
        "/mlcv2/WorkingSpace/Personal/baotg/TTam/you_know/source/visualize"
    ) / ("fusion_multi_model_" + args.sub_folder)
    (folder_visualize_multi_model).mkdir(exist_ok=True)

    folder_test = Path(
        "/mlcv2/WorkingSpace/Personal/baotg/TTam/you_know/dataset/data_private/images"
    )

    list_images = list(folder_test.iterdir())
    list_data_boxes, list_data_scores = [], []

    for results_model in tqdm(args.file_json):
        with open(results_model, "r") as f:
            data = json.load(f)

        dict_data_boxes, dict_data_scores = {}, {}
        for item in data:

            if item["image_name"] not in dict_data_boxes:
                dict_data_boxes[item["image_name"]] = []
                dict_data_boxes[item["image_name"]].append(item["bbox"])
                dict_data_scores[item["image_name"]] = [item["score"]]
            else:
                dict_data_boxes[item["image_name"]].append(item["bbox"])
                dict_data_scores[item["image_name"]].extend([item["score"]])

        list_data_boxes.append(dict_data_boxes)
        list_data_scores.append(dict_data_scores)

    with open(file_txt, "a+") as f:
        for key in tqdm(list_images):
            key = key.stem

            boxes_list, scores_list = [], []

            for model in range(len(list_data_boxes)):
                try:
                    boxes = list_data_boxes[model][key]
                    scores = list_data_scores[model][key]

                except:
                    print(f"Error: {key} not found in model {model}")
                    boxes = []
                    scores = []

                boxes_list.append(boxes)
                scores_list.append(scores)
            
            file_image = folder_test / key
            file_multi_models = folder_visualize_multi_model / key

            visualize(
                (str(file_image) + ".jpg"),
                boxes_list,
                scores_list,
                (str(file_multi_models) + ".jpg"),
            )

            labels_list = [[0] * len(boxes) for boxes in boxes_list]

            if args.method == "nms":
                boxes, scores, labels = nms(
                    boxes_list,
                    scores_list,
                    labels_list,
                    weights=args.weights,
                    iou_thr=args.iou_thr,
                )
            elif args.method == "soft_nms":
                boxes, scores, labels = soft_nms(
                    boxes_list,
                    scores_list,
                    labels_list,
                    weights=args.weights,
                    iou_thr=args.iou_thr,
                    sigma=args.sigma,
                    thresh=args.skip_box_thr,
                )
            elif args.method == "non_maximum_weighted":
                boxes, scores, labels = non_maximum_weighted(
                    boxes_list,
                    scores_list,
                    labels_list,
                    weights=args.weights,
                    iou_thr=args.iou_thr,
                    skip_box_thr=0.0001,
                )
            elif args.method == "weighted_boxes_fusion":
                weights = args.weights
                # if key in "test":
                #     weights = [2, 1, 1]
                # # else:
                    # weights = [0, 1, 1]
                boxes, scores, labels = weighted_boxes_fusion(
                    boxes_list,
                    scores_list,
                    labels_list,
                    weights=weights,
                    iou_thr=0.5,
                    skip_box_thr=0.0001,
                    conf_type="box_and_model_avg"
                )

            # check_missing_boxes(boxes, scores, labels, boxes_list, scores_list, labels_list)

            file_image = folder_test / key
            file_visualize = folder_visualize / key

            visualize(
                (str(file_image) + ".jpg"),
                [boxes],
                [scores],
                (str(file_visualize) + ".jpg"),
            )
            
            for box, score in zip(boxes, scores):
                if score < args.skip_box_thr:
                    continue

                x_min, y_min, x_max, y_max = box
                if math.isnan(x_min) or math.isnan(y_min) or math.isnan(x_max) or math.isnan(y_max) or math.isnan(score):
                    continue
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                w = x_max - x_min
                h = y_max - y_min
                f.write(f"{key} 0 {x_center} {y_center} {w} {h}\n")

    with zipfile.ZipFile(sub_folder / (args.sub_folder + ".zip"), "w") as f:
        f.write(file_txt, "answer.txt")


if __name__ == "__main__":
    main()

"""python source/tools/fusion.py --file-json submissions/yolov11/results.json submissions/yolov7/results.json submissions/dfine/results.json"""


"""python source/tools/fusion.py --file-json submissions/dfinel_augement_train_test_img_best_1/results.json submissions/dfine_augement_train_0.2_conf/results.json submissions/dfine_best_0.2_conf/results.json --sub-folder 3_dfine_augement_train_best_1_new --weight 1 1 1"""
