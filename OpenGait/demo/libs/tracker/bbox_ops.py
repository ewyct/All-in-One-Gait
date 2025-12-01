"""
Pure Python/NumPy replacement for cython_bbox.bbox_overlaps
"""
import numpy as np


def bbox_overlaps(boxes1, boxes2):
    """
    Compute IoU between two sets of bounding boxes.

    Args:
        boxes1: (N, 4) array of boxes in format [x1, y1, x2, y2]
        boxes2: (M, 4) array of boxes in format [x1, y1, x2, y2]

    Returns:
        (N, M) array of IoU values
    """
    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)

    N = boxes1.shape[0]
    M = boxes2.shape[0]

    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)

    # Compute areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute intersection
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    wh = np.clip(rb - lt, 0, None)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    # Compute union and IoU
    union = area1[:, None] + area2 - inter
    iou = inter / np.clip(union, 1e-6, None)

    return iou.astype(np.float64)
