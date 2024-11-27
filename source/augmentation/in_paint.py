import sys
import random

from tqdm import tqdm
from pathlib import Path
from typing import Dict, List
from PIL import Image, ImageDraw

# sys.path.insert(0, str(Path(__file__).parent.parent))
# sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from schemas import AnnotationData, ImageData
from utils.bbox import calculate_iou, calculate_overlap


def group_images_by_camera_id(
    data: List[ImageData],
) -> Dict[str, List[ImageData]]:
    """
    Group images by camera id.

    Args:
        data (List[ImageData]): List of images to group by camera id

    Returns:
        Dict: Dictionary containing camera id as key and list of images as value
    """

    unique_camera_ids = set(["_".join(image.name.split("_")[:2]) for image in data])
    grouped_images = {camera_id: [] for camera_id in unique_camera_ids}

    for camera_id in unique_camera_ids:
        for image in data:
            if image.name.startswith(camera_id):
                grouped_images[camera_id].append(image)

    return grouped_images


def augment_in_paint(data: List[ImageData], max_generations=5) -> List[ImageData]:
    """
    Augment data using in-paint strategy. https://arxiv.org/pdf/2404.11226

    Args:
        data (List[ImageData]): List of images to augment

    Returns:
        List[ImageData]: List of augmented images
    """

    results: List[ImageData] = []

    updated_images = []

    for image in data:
        annotations = []
        for annotation in image.annotations:
            annotations.append(
                AnnotationData(
                    class_id=annotation.class_id,
                    bbox=annotation.bbox,
                    cropped_image=image.image.crop(annotation.bbox),
                )
            )

        updated_images.append(
            ImageData(
                name=image.name,
                image=image.image,
                width=image.width,
                height=image.height,
                annotations=annotations,
            )
        )

    images = updated_images
    annotations = [image.annotations for image in images]

    for image in tqdm(images):
        random.shuffle(annotations)
        grouped_bboxes: List[AnnotationData] = []

        for image_annotation in annotations:
            for bbox in image_annotation:
                grouped_bboxes.append(bbox)

        image_annotations = image.annotations  # annotations of current image
        generated_annotations = {0: 0}

        original_image_bboxes = [bbox for bbox in image_annotations]
        added_bboxes = []

        for bbox_of_group in grouped_bboxes:
            if generated_annotations[bbox_of_group.class_id] == max_generations:
                break

            should_add = True

            for bbox_of_image in original_image_bboxes:
                # if bbox_of_group.class_id != bbox_of_image.class_id:
                #     continue
                iou_overlap = calculate_overlap(
                    bbox_of_group.bbox, bbox_of_image.bbox
                )

                if iou_overlap > 0.05:
                    should_add = False

            if should_add:
                generated_annotations[bbox_of_group.class_id] += 1
                original_image_bboxes.append(bbox_of_group)
                added_bboxes.append(bbox_of_group)

        updated_image = image.image.copy()
        for bbox in added_bboxes:
            updated_image.paste(
                bbox.cropped_image, (int(bbox.bbox[0]), int(bbox.bbox[1]))
            )

        results.append(
            ImageData(
                name=image.name,
                image=updated_image,
                width=image.width,
                height=image.height,
                annotations=original_image_bboxes,
            )
        )

            # visualize the augmented image
            # img = Path(
            #     "/mlcv2/WorkingSpace/Personal/baotg/BKAI/Vehicle/data/train/daytime"
            # ) / (image.name + ".jpg")

            # img = Image.open(img)
            # draw = ImageDraw.Draw(img)

            # for bbox in original_image_bboxes:
            #     if bbox in added_bboxes:
            #         img.paste(
            #             bbox.cropped_image, (int(bbox.bbox[0]), int(bbox.bbox[1]))
            #         )

            #         draw.rectangle(bbox.bbox, outline="green")

            #     else:
            #         draw.rectangle(bbox.bbox, outline="red")

            # img.save(
            #     Path(
            #         "/mlcv2/WorkingSpace/Personal/baotg/BKAI/Vehicle/source/data-augmentation/augmented"
            #     )
            #     / (image.name + ".jpg")
            # )

    return results

def augment_in_paint_synthetic(data_ori: List[ImageData], data_synthetic: List[ImageData], max_generations=5) -> List[ImageData]:
    """
    Augment data using in-paint strategy. https://arxiv.org/pdf/2404.11226

    Args:
        data (List[ImageData]): List of images to augment
        data_synthetic (List[ImageData]): List of synthetic images to augment

    Returns:
        List[ImageData]: List of augmented images
    """

    results: List[ImageData] = []

    updated_images_ori, updated_images_synthetic = [], []

    for image in tqdm(data_ori, desc="Loading data ori"):
        annotations = []
        for annotation in image.annotations:
            
            w = int((annotation.bbox[2] - annotation.bbox[0]) / 1.5)
            h = int((annotation.bbox[3] - annotation.bbox[1]) / 1.5)

            cropped_image = image.image.crop(annotation.bbox).resize([w, h])

            annotation.bbox[2] = annotation.bbox[0] + w
            annotation.bbox[3] = annotation.bbox[1] + h
            
            annotations.append(
                AnnotationData(
                    class_id=annotation.class_id,
                    bbox=annotation.bbox,
                    cropped_image=cropped_image,
                )
            )

        updated_images_ori.append(
            ImageData(
                name=image.name,
                image=image.image,
                width=image.width,
                height=image.height,
                annotations=annotations,
            )
        )

    for image in tqdm(data_synthetic, desc="Loading data synthetic"):
        annotations = []
        for annotation in image.annotations:
            annotations.append(
                AnnotationData(
                    class_id=annotation.class_id,
                    bbox=annotation.bbox,
                    cropped_image=image.image.crop(annotation.bbox),
                )
            )

        updated_images_synthetic.append(
            ImageData(
                name=image.name,
                image=image.image,
                width=image.width,
                height=image.height,
                annotations=annotations,
            )
        )

    images = updated_images_ori
    images_synthetic = updated_images_synthetic
    annotations = [image.annotations for image in images]
    for image in images:
        if image.name.split("_")[0] == "test":
            annotations.append(image.annotations)
            annotations.append(image.annotations)

    for image in tqdm(images_synthetic, desc="Augmenting synthetic data"):
        random.shuffle(annotations)
        grouped_bboxes: List[AnnotationData] = []

        for image_annotation in annotations:
            for bbox in image_annotation:
                grouped_bboxes.append(bbox)

        image_annotations = image.annotations  # annotations of current image
        generated_annotations = {0: 0}

        original_image_bboxes = [bbox for bbox in image_annotations]
        added_bboxes = []

        for bbox_of_group in grouped_bboxes:
            if generated_annotations[bbox_of_group.class_id] == max_generations:
                break

            should_add = True

            # Check IOU with all existing bounding boxes (original and added)
            for bbox_of_image in original_image_bboxes + added_bboxes:
                iou_overlap = calculate_overlap(
                    bbox_of_group.bbox, bbox_of_image.bbox
                )

                if iou_overlap > 0.05:
                    should_add = False
                    break

            # Ensure bounding box does not go out of bounds
            if bbox_of_group.bbox[2] > image.width or bbox_of_group.bbox[3] > image.height or bbox_of_group.bbox[0] < 0 or bbox_of_group.bbox[1] < 0:
                should_add = False

            if should_add:
                generated_annotations[bbox_of_group.class_id] += 1
                original_image_bboxes.append(bbox_of_group)
                added_bboxes.append(bbox_of_group)

        updated_image = image.image.copy()
        for bbox in original_image_bboxes:
            updated_image.paste(
                bbox.cropped_image, (int(bbox.bbox[0]), int(bbox.bbox[1]))
            )
            
        results.append(
            ImageData(
                name=image.name,
                image=updated_image,
                width=image.width,
                height=image.height,
                annotations=original_image_bboxes,
            )
        )

    return results
