import json
import math
import zipfile
import argparse

from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--images", type=str, required=False, default="dataset/data_private/images", help="Path to the images")
parser.add_argument("--sub-folder", type=str, required=True, default="yolov11", help="Submission folder")
parser.add_argument("--model", type=str, required=False, default="yolov11", help="Model")
parser.add_argument("--threshold", type=float, required=False, default=0.4, help="Threshold")
parser.add_argument("--batch-size", type=int, required=False, default=50, help="Batch size")
args = parser.parse_args()

path_images = Path(args.images)
name_folder = Path("./submissions") / args.sub_folder

(name_folder).mkdir(exist_ok=True)

if args.model == "yolov11":
    model = YOLO("/mlcv2/WorkingSpace/Personal/baotg/TTam/you_know/models/yolo/runs/detect/train3/weights/best.pt")
elif args.model == "yolov8":
    model = YOLO("/mlcv2/WorkingSpace/Personal/baotg/TTam/you_know/models/yolov8/best_performance/train3_finetune_l_640/weights/best.pt")
    # yolov8l 150ep
    # model = YOLO("/mlcv2/WorkingSpace/Personal/baotg/TTam/you_know/models/yolov8/runs/detect/train/weights/best.pt")
    # yolov8l 200ep
    # model = YOLO("/mlcv2/WorkingSpace/Personal/baotg/TTam/you_know/models/yolov8/runs/detect/train3_finetune_l_640/weights/best.pt")
    # yolov8l finetune augement data
    # model = YOLO("/mlcv2/WorkingSpace/Personal/baotg/TTam/you_know/models/yolov8/runs/detect/train3/weights/best.pt")
    # yolov8l finetune augement 50ep - best
    # model = YOLO("/mlcv2/WorkingSpace/Personal/baotg/TTam/you_know/models/yolov8/runs/detect/train5/weights/best.pt")
    # yolov8x 300ep
    # model = YOLO("/mlcv2/WorkingSpace/Personal/baotg/TTam/you_know/models/yolov8/runs/detect/train_x_640/weights/best.pt")
    # yolov8x 450ep
    # model = YOLO("/mlcv2/WorkingSpace/Personal/baotg/TTam/you_know/models/yolov8/runs/detect/train3_next_x_640/weights/best.pt")
else:
    raise ValueError("Model not found")

list_imgs = list(path_images.iterdir())

print("len imgs: ", len(list_imgs))


results_bbox_scores = []
for i in range(math.ceil(len(list_imgs) / args.batch_size)):
    list_imgs_batch_size = list_imgs[i * args.batch_size : (i + 1) * args.batch_size]
    results = model(list_imgs_batch_size, augment=True)

    for result, img_path in tqdm(zip(results, list_imgs_batch_size), total=len(list_imgs_batch_size), desc=f"Batch {i}"):
        boxes = result.boxes
        img = Image.open(img_path)
        width, height = img.size
        # img1 = ImageDraw.Draw(img)

        with open(name_folder / ("answer.txt"), "a+") as file_labels:
            for bbox in boxes:
                x, y, w, h = map(float, bbox.xywh[0].tolist())
                score = bbox.conf.item()
                
                if score < args.threshold:
                    continue
                
                x /= width
                y /= height
                w /= width
                h /= height
                
                x_min = x - w / 2
                y_min = y - h / 2
                x_max = x + w / 2
                y_max = y + h / 2
                
                file_labels.write(img_path.stem + " 0 " + " ".join(map(str, [x, y, w, h])) + "\n")
                # print(img_path.stem + " 0 " + " ".join(map(str, [x, y, w, h])) + "\n")
                results_bbox_scores.append({"image_name": img_path.stem, "bbox": [x_min, y_min, x_max, y_max], "score": score})

with open (name_folder / "results.json", "w") as f:
    json.dump(results_bbox_scores, f)

with zipfile.ZipFile(name_folder / (args.sub_folder + ".zip") , "w") as f:
    f.write(name_folder / "answer.txt", "answer.txt")


"""CUDA_VISIBLE_DEVICES=4 python source/tools/inference_yolo.py --sub-folder yolov11"""
