import sys
from PIL import Image
from tqdm import tqdm
from typing import List
from pathlib import Path
from argparse import ArgumentParser

sys.path.insert(0, str(Path(__file__).parent.parent))
from schemas import AnnotationData, ImageData
from augmentation import augment_in_paint, augment_in_paint_synthetic

# sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.bbox import convert_yolo_to_xyxy, convert_xyxy_to_yolo


def parse_args():
    """
    Args:
        input (str): The folder containing the images and annotations in YOLO format
        output (str): The folder to save the images and annotations in COCO format
    """

    parser = ArgumentParser(description="Data augmentation for in-paint")
    parser.add_argument("--input", type=str, required=True, help="Input image")
    parser.add_argument("--input-synthetic", type=str, required=True, help="Input image synthetic")
    parser.add_argument(
        "--strategy", type=str, required=True, choices=["in-paint"], default="in-paint", help="Strategy"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output image after augmentation"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    original_input_path = Path(args.input)
    original_input_synthetic_path = Path(args.input_synthetic)
    original_output_path = Path(args.output)
    strategy = args.strategy
    
    original_output_path.mkdir(parents=True, exist_ok=True)

    # load data
    data: List[ImageData] = []
    data_synthetic: List[ImageData] = []

    for image_path in tqdm(
        list(original_input_path.iterdir()), desc="Loading data"
    ):
        # if image_path.suffix not in [".png"]:
            # continue
        image = Image.open(image_path)
        width, height = image.size

        # label_path = image_path.with_suffix(".txt")
        label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        annotations = label_path.read_text().strip().split("\n")
        annotations = [ann.strip() for ann in annotations if ann.strip()]
        annos: List[AnnotationData] = []

        for annotation in annotations:
            class_id, x_center, y_center, w, h = map(float, annotation.split())

            class_id = int(class_id)
            x_min, y_min, x_max, y_max = convert_yolo_to_xyxy(
                [x_center, y_center, w, h], (width, height)
            )

            annos.append(
                AnnotationData(
                    class_id=class_id,
                    bbox=[x_min, y_min, x_max, y_max],
                    cropped_image=None,
                )
            )

        data.append(
            ImageData(
                name=image_path.stem,
                image=image,
                width=width,
                height=height,
                annotations=annos,
            )
        )
        
    for image_path in tqdm(
        list(original_input_synthetic_path.iterdir()), desc="Loading data"
    ):
        image = Image.open(image_path)
        image = image.resize((1000, 1000))
        width, height = image.size

        # label_path = image_path.with_suffix(".txt")
        label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        annotations = label_path.read_text().strip().split("\n")
        annotations = [ann.strip() for ann in annotations if ann.strip()]
        annos: List[AnnotationData] = []

        data_synthetic.append(
            ImageData(
                name=image_path.stem,
                image=image,
                width=width,
                height=height,
                annotations=annos,
            )
        )

    if strategy == "in-paint":
        print("Augmenting data using in-paint strategy")

        # augmented_data = augment_in_paint(data)
        augmented_data = augment_in_paint_synthetic(data, data_synthetic)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    (original_output_path / "images").mkdir(parents=True, exist_ok=True)
    (original_output_path / "labels").mkdir(parents=True, exist_ok=True)

    for augmendted_image in augmented_data:
        image_path = original_output_path / "images" / f"{augmendted_image.name}.jpg"
        label_path = original_output_path / "labels" / f"{augmendted_image.name}.txt"

        augmendted_image.image.save(image_path)
        output = ""
        for ann in augmendted_image.annotations:
            bbox = ann.bbox
            bbox = convert_xyxy_to_yolo(
                bbox, (augmendted_image.width, augmendted_image.height)
            )
            
            output += f"{ann.class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"

        label_path.write_text(output.strip())


if __name__ == "__main__":
    main()
    pass
