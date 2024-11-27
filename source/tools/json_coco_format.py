import sys
import json
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from source.utils import generate_coco_annotations

parser = argparse.ArgumentParser(description="Convert YOLO format to COCO format")
parser.add_argument(
    "--images",
    type=Path,
    required=True,
    help="Path to the folder containing images and labels",
)
parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Path to the output json file",
)
args = parser.parse_args()

path_images = args.images

def main():
    list_images = list(path_images.iterdir())
    with open(args.output, "w") as f:
        json.dump(generate_coco_annotations(list_images), f)
   
if __name__ == "__main__":
    main()
