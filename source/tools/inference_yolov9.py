import sys
import cv2
import json
import torch
import zipfile
import argparse

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from numpy import ndarray
from typing import Dict, List, Tuple, Any

from ultralytics import YOLO

sys.path.insert(
    0, str(Path(__file__).absolute().parent.parent.parent / "models" / "yolov9")
)

print(sys)
from models.common import DetectMultiBackend
from utils.dataloaders import (
    IMG_FORMATS,
    VID_FORMATS,
    LoadImages,
    LoadScreenshots,
    LoadStreams,
)
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folder",
        default="/mlcv2/WorkingSpace/Personal/baotg/TTam/you_know/dataset/data_test/images",
        help="Image file",
    )
    parser.add_argument("--checkpoint", help="Checkpoint file")
    parser.add_argument("--visualize-out", default=None)
    parser.add_argument("--file-out", default=None, help="Path to output file")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--score-thr", type=float, default=0.4)

    # CODETR params
    parser.add_argument("--config", help="Config file")

    # YOLO params
    parser.add_argument("--img-size", type=int, default=640)

    # YOLOv7 params
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--trace", action="store_true")

    args = parser.parse_args()
    return args


def convert_to_yolo_format(bbox: List[float], img_shape: Tuple[int, int]):
    img_h, img_w = img_shape
    x1, y1, x2, y2 = bbox

    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    w = x2 - x1
    h = y2 - y1

    x_center /= img_w
    y_center /= img_h
    w /= img_w
    h /= img_h

    return x_center, y_center, w, h


def infer_yolov9(args, images: Dict[str, ndarray]):
    device = torch.device("cuda")
    model = DetectMultiBackend(
        args.checkpoint, device=device, data=args.config, fp16=True
    )
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(args.img_size, s=stride)  # check image size

    bs = 1
    dataset = LoadImages(args.folder, img_size=imgsz, stride=stride, auto=pt)
    model.warmup(
        imgsz=(1 if pt or model.triton else bs, 3, *(args.img_size, args.img_size))
    )  # warmup
    results = []

    for path, im, im0s, vid_cap, s in tqdm(dataset):
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(im, augment=args.augment)[0]

        # NMS
        pred = non_max_suppression(
            pred, args.score_thr, args.iou_thres, [0, 1, 2, 3], False, max_det=300
        )

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)
            p = Path(p)  # to Path
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    xywh = (
                        (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                        .view(-1)
                        .tolist()
                    )  # normalized xywh

                    results.append(
                        {
                            "image_id": p.name,
                            "category_id": cls,
                            "bbox": xywh,
                            "score": conf.item(),
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
    Path(args.visualize_out).mkdir(exist_ok=True, parents=True)

    used_images = set()

    for result in tqdm(results):
        image_id = result["image_id"]

        if image_id in used_images:
            continue

        img = cv2.imread(Path(args.folder) / result["image_id"])

        assert img is not None, f"Image not found: {result['image_id']}"

        for result in results:
            if result["image_id"] == image_id:
                used_images.add(image_id)

                x_center, y_center, w, h = result["bbox"]
                x1 = int((x_center - w / 2) * img.shape[1])
                y1 = int((y_center - h / 2) * img.shape[0])
                x2 = int((x_center + w / 2) * img.shape[1])
                y2 = int((y_center + h / 2) * img.shape[0])

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(
                    img,
                    f"{image_id} {result['score']:.2f}",
                    # f"{0}_{result['score']:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2,
                )

        cv2.imwrite(Path(args.visualize_out) / image_id, img)


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

            results_bbox_scores.append(
                {
                    "image_name": Path(image_id).stem,
                    "bbox": [x_min, y_min, x_max, y_max],
                    "score": float(score),
                }
            )

    with zipfile.ZipFile(path_folder / (str(sub_folder) + ".zip"), "w") as f:
        f.write(path_folder / "answer.txt", "answer.txt")

    with open(path_folder / "results.json", "w") as f:
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

        # process = {}
        # for i in range(num_workers):
        #     current_l_images = l_images[i*len(l_images)//num_workers:(i+1)*len(l_images)//num_workers]
        #     process[i] = multiprocessing.Process(target=lambda x: {Path(img_path).name: cv2.imread(str(img_path)) for img_path in x}, args=(current_l_images,))

    # inference

    results = infer_yolov9(args, images)

    assert len(results) > 0, "No results found"

    # save visualization
    if args.visualize_out:
        print("Visualizing results...")
        visualize_results(args, results)

    # save results
    if args.file_out:
        print("Saving results...")
        save_submission(args.file_out, results)


if __name__ == "__main__":
    main()
