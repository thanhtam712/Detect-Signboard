import argparse

from pathlib import Path

parser = argparse.ArgumentParser("folder labels")
parser.add_argument("--images-folder", type=str, required=False, default="dataset/data_test/images", help="Images folder")
parser.add_argument("--labels-file", type=str, required=False, default="dataset/data_test/labels/answer.txt", help="Labels file")
parser.add_argument("--output-folder", type=str, required=False, default="dataset/data_test/labels", help="Output folder")
args = parser.parse_args()

path_images_folder = Path(args.images_folder)
path_labels_file = Path(args.labels_file)
path_output_folder = Path(args.output_folder)

with open(path_labels_file, "r") as f:
    lines = f.readlines()
    for line in lines:
        img_name = line.strip().split(" ")[0]
                
        with open(path_output_folder / (img_name + ".txt"), "a") as f_out:
            f_out.write(line.strip() + "\n")


for img_path in path_images_folder.iterdir():
    img_name = img_path.stem
    if not (path_output_folder / (img_name + ".txt")).exists():
        with open(path_output_folder / (img_name + ".txt"), "w") as f_out:
            f_out.write("")
