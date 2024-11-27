from typing import List, Tuple


def convert_xyxy_to_yolo(
    bbox: List[float], img_size: Tuple[int, int]
) -> Tuple[float, float, float, float]:
    """
    Convert bounding box from [x_min, y_min, x_max, y_max] format
    to [x_center, y_center, w, h] format

    Args:
        bbox: bounding box in [x_min, y_min, x_max, y_max] format
        img_size: image size in (width, height) format

    Returns:
        Tuple of bounding box in [x_center, y_center, w, h] format
    """

    img_w, img_h = img_size
    x_min, y_min, x_max, y_max = bbox
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    w = x_max - x_min
    h = y_max - y_min

    x_center /= img_w
    y_center /= img_h
    w /= img_w
    h /= img_h

    return x_center, y_center, w, h


def convert_yolo_to_xyxy(
    bbox: List[float], img_size: Tuple[int, int]
) -> Tuple[float, float, float, float]:
    """
    Convert bounding box from [x_center, y_center, w, h] format
    to [x_min, y_min, x_max, y_max] format

    Args:
        bbox: bounding box in [x_center, y_center, w, h] format
        img_size: image size in (width, height) format

    Returns:
        Tuple of bounding box in [x_min, y_min, x_max, y_max] format
    """

    img_w, img_h = img_size
    x_center, y_center, w, h = bbox

    x_center *= img_w
    y_center *= img_h
    w *= img_w
    h *= img_h

    x_min = x_center - w / 2
    y_min = y_center - h / 2
    x_max = x_center + w / 2
    y_max = y_center + h / 2

    return x_min, y_min, x_max, y_max


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes

    Args:
        box1: bounding box in [x_min, y_min, x_max, y_max] format
        box2: bounding box in [x_min, y_min, x_max, y_max] format

    Returns:
        IoU value
    """

    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)

    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou


def calculate_overlap(box1: List[float], box2: List[float]) -> float:
    """
    Calculate overlap between two bounding boxes

    Args:
        box1: bounding box in [x_min, y_min, x_max, y_max] format
        box2: bounding box in [x_min, y_min, x_max, y_max] format

    Returns:
        Overlap value
    """

    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)

    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)

    if min(box1_area, box2_area) != 0:
        overlap = inter_area / min(box1_area, box2_area)
    else:
        overlap = 0
    return overlap
