import os
import cv2
import json
import zipfile
import argparse

from tqdm import tqdm
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-json", nargs="+", help="Path to json file")
    parser.add_argument("--sub-folder", type=str, required=True, default="fusion", help="Path to output file")
    parser.add_argument("--iou-thr", type=float, default=0.1)
    parser.add_argument("--skip-box-thr", type=float, default=0.4)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--weights", nargs="+", type=float, default=[1, 1])
    parser.add_argument("--method", type=str, default="nms")
    parser.add_argument("--output", type=str, default="output.json")

    return parser.parse_args()

def iou_thr(boxA: list[float], boxB: list[float]) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        boxA: list of 4 float values [x_min, y_min, x_max, y_max]
        boxB: list of 4 float values [x_min, y_min, x_max, y_max]
    
    Returns:
        iou: float
        boxAArea: float
        boxBArea: float
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou, boxAArea, boxBArea


def visualize(image: str, list_boxes: list[float], list_scores: float, output: str):
    """
    Visualize the bounding boxes on the image
    
    Args:
        image: str, path to the image
        list_boxes: list of list of float, list of bounding boxes
        list_scores: list of float, list of scores
        output: str, path to the output image
    """
    
    args = parse_args()
    
    if os.path.exists(output):
        print(f"Error: The file {output} already exists.")
        return
    
    if not os.path.exists(image):
        image = image.replace(".jpg", ".png")
        if not os.path.exists(image):
            print(f"Error: The file {image} does not exist.")
            return

    colors = [(0, 0, 255), (0, 255, 0)]
    models = ["yolov8", "dfine"]

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
            x_min, y_min, x_max, y_max = int(x_min * width), int(y_min * height), int(x_max * width), int(y_max * height)
            
            img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), colors[idx], 2)
            img = cv2.putText(img, f"{score:.2f} {models[idx]}", (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[idx], 2)

    cv2.imwrite(output, img)


def main():
    args = parse_args()
    
    sub_folder = Path("./submissions") / ("fusion_" + args.sub_folder)
    (sub_folder).mkdir(exist_ok=True)
    file_txt = sub_folder / "answer.txt"

    folder_visualize = Path("/mlcv2/WorkingSpace/Personal/baotg/TTam/you_know/source/visualize") / ("fusion_" + args.sub_folder)
    (folder_visualize).mkdir(exist_ok=True)
    
    folder_visualize_multi_model = Path("/mlcv2/WorkingSpace/Personal/baotg/TTam/you_know/source/visualize") / ("fusion_multi_model_" + args.sub_folder)
    (folder_visualize_multi_model).mkdir(exist_ok=True)

    folder_test = Path("/mlcv2/WorkingSpace/Personal/baotg/TTam/you_know/dataset/data_test/images")
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
    
    result_bbox_scores, result_bbox = [], []
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
            
            visualize((str(file_image) + ".jpg"), boxes_list, scores_list, (str(file_multi_models) + ".jpg"))
            
            for box, score in zip(boxes_list[1], scores_list[1]):
                for box2, score2 in zip(boxes_list[0], scores_list[0]):
                    iou, boxAArea, boxBArea = iou_thr(box, box2)
                    if iou > 0.5:
                        if boxAArea > boxBArea:
                            if box not in result_bbox:
                                result_bbox.append(box)
                                result_bbox_scores.append(score)
                                boxes_list[0].remove(box2)
                                boxes_list[1].remove(box)
                                scores_list[0].remove(score2)
                                scores_list[1].remove(score)
                        else:
                            if box2 not in result_bbox:
                                result_bbox.append(box2)
                                result_bbox_scores.append(score2)
                            boxes_list[1].remove(box)
                            boxes_list[0].remove(box2)
                            scores_list[1].remove(score)
                            scores_list[0].remove(score2)
                        
                        break
                                                

            for box, score in zip(boxes_list[0], scores_list[0]):
                result_bbox.append(box)
                result_bbox_scores.append(score)
                
            for box, score in zip(boxes_list[1], scores_list[1]):
                result_bbox.append(box)
                result_bbox_scores.append(score)
                        
            file_image = folder_test / key 
            file_visualize = folder_visualize / key
            
            visualize((str(file_image) + ".jpg"), [boxes], [scores], (str(file_visualize) + ".jpg"))
                
            for box, score in zip(boxes, scores):
                if score < args.skip_box_thr:
                    continue
                
                x_min, y_min, x_max, y_max = box
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
