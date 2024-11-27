import random
import shutil
import argparse

from pathlib import Path

parser = argparse.ArgumentParser("random data")
parser.add_argument("--original-data", type=str, required=True, help="Original folder data")
parser.add_argument("--train-data", type=str, required=True, help="Train folder data")
parser.add_argument("--val-data", type=str, required=True, help="Val folder data")
args = parser.parse_args()

path_original_data = Path(args.original_data)
path_train_data = Path(args.train_data)
path_val_data = Path(args.val_data)

path_train_data.mkdir(parents=True, exist_ok=True)
path_val_data.mkdir(parents=True, exist_ok=True)

(path_train_data / "images").mkdir(parents=True, exist_ok=True)
(path_train_data / "labels").mkdir(parents=True, exist_ok=True)
(path_val_data / "images").mkdir(parents=True, exist_ok=True)
(path_val_data / "labels").mkdir(parents=True, exist_ok=True)

train_ratio = 0.9

def main():
    list_imgs_ori = list(Path(path_original_data / "images").iterdir())
    list_imgs_ori_IMG = [img for img in list_imgs_ori if img.suffix == ".jpg"]
    list_imgs_ori_test = [img for img in list_imgs_ori if img.suffix == ".png"]
    
    random.shuffle(list_imgs_ori_IMG)
    
    train_imgs = list_imgs_ori_IMG[:int(len(list_imgs_ori_IMG) * train_ratio)]
    train_imgs += list_imgs_ori_test
    val_imgs = list_imgs_ori_IMG[int(len(list_imgs_ori_IMG) * train_ratio):]
    
    for img in train_imgs:
        dst_path = path_train_data / "images" / img.name 
        label_src_path = path_original_data / "labels" / (img.stem + ".txt")
        label_dst_path = path_train_data / "labels" / (img.stem + ".txt")
        shutil.copy(img, dst_path)
        shutil.copy(label_src_path, label_dst_path)
        
    for img in val_imgs:
        dst_path = path_val_data / "images" / img.name
        label_src_path = path_original_data / "labels" / (img.stem + ".txt")
        label_dst_path = path_val_data / "labels" / (img.stem + ".txt")
        shutil.copy(img, dst_path)
        shutil.copy(label_src_path, label_dst_path)
        
if __name__ == "__main__":
    main()
        
        
