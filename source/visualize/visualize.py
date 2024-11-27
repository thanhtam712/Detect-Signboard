import argparse

from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument("--images", type=str, required=False, default="/mlcv2/WorkingSpace/Personal/baotg/TTam/you_know/dataset/data_test/images", help="Path to the images")
parser.add_argument("--labels", type=str, required=True, help="Path to the labels file")
parser.add_argument("--output-dir", type=str, required=False, default="/mlcv2/WorkingSpace/Personal/baotg/TTam/you_know/source/visualize/output", help="Path to the output directory")
args = parser.parse_args()


path_images = Path(args.images)
path_labels = Path(args.labels)
output_dir = Path(args.output_dir)

output_dir.mkdir(exist_ok=True)

with open(path_labels, "r") as f:
    for line in tqdm(f):
        image_name = line.strip().split()[0]
        class_id, x_center, y_center, w, h = map(float, line.strip().split()[1:])
        output_path = output_dir / f"{image_name}_bbox.jpg"


        
        if not output_path.exists():
            try:
                image_path = path_images / f"{image_name}.jpg"
                image = Image.open(image_path)
                
            except:
                image_path = path_images / f"{image_name}.png"
                image = Image.open(image_path)
        else:
            image = Image.open(output_path)
        draw = ImageDraw.Draw(image)

        x_min = int((x_center - w / 2) * image.width)
        y_min = int((y_center - h / 2) * image.height)
        x_max = int((x_center + w / 2) * image.width)
        y_max = int((y_center + h / 2) * image.height)

        draw.rectangle([x_min, y_min, x_max, y_max], outline="red")

        image.save(output_path)
        print(f"Saved {output_path}")


# for img in list(path_images.iterdir()):
#     label_path = path_labels / f"{img.stem}.txt"
    
#     image_name = img.name

#     image_path = path_images / f"{image_name}"
#     image = Image.open(image_path)
#     draw = ImageDraw.Draw(image)

#     output_path = output_dir / f"{image_name}_bbox.jpg"

#     with open(label_path, "r") as f:
#         for line in f:    
#             class_id, x_center, y_center, w, h = map(float, line.strip().split())

#             x_min = int((x_center - w / 2) * image.width)
#             y_min = int((y_center - h / 2) * image.height)
#             x_max = int((x_center + w / 2) * image.width)
#             y_max = int((y_center + h / 2) * image.height)

#             draw.rectangle([x_min, y_min, x_max, y_max], outline="red")

#     image.save(output_path)
#     print(f"Saved {output_path}")
