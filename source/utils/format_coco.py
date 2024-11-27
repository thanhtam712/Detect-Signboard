import argparse

from PIL import Image
from tqdm import tqdm
from pathlib import Path

from .bbox import convert_yolo_to_xyxy

def generate_coco_annotations(
    img_paths: list[Path], offset: int = 0
):
    """
    Generate COCO annotations from the given image paths
    In the input image paths, the corresponding label files are in the same folder
    """

    images = []
    annotations = []
    categories = []

    for index_image, img_path in tqdm(enumerate(img_paths)):
        img = Image.open(img_path)
        label_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")

        images.append(
            {
                "id": index_image + offset,
                "file_name": img_path.name,
                "height": img.size[1],
                "width": img.size[0],
            }
        )

        for annotation in open(label_path, "r").readlines():
            annotation = annotation.strip()

            if annotation == "":
                continue

            class_label, x_center, y_center, width, height = map(
                float, annotation.split()
            )

            x_min, y_min, x_max, y_max = convert_yolo_to_xyxy(
                [x_center, y_center, width, height],
                (
                    img.size[0],
                    img.size[1],
                ),
            )
            class_label = 0
            annotations.append(
                {
                    "id": len(annotations) + offset,
                    "image_id": index_image + offset,
                    "category_id": class_label,
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )

    categories = [
        {
            "id": 0,
            "name": "signboard",
            "supercategory": "signboard",
        },
    ]

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
