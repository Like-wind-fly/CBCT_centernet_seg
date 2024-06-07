import torch
from ..stats_logger import StatsLogger
from .SGA_mertics import SGA_dsc,evaluate_mesh as SGA_evaluate_mesh
from neural_parts_code.metrics.detection_mertics import object_detection_metrics


def get_metrics(keys):
    metrics = {
        "detection_metrics":object_detection_metrics,
        "SGA_dsc": SGA_dsc,
        "SGA_evaluate_mesh": SGA_evaluate_mesh
    }

    if keys is None:
        keys = []
    return list_of_metrics(*[metrics[k] for k in keys])


def list_of_metrics(*functions):
    def inner(*args, **kwargs):
        return [
            metric_fn(*args, **kwargs) for metric_fn in functions
        ]
    return inner

def none(predictions, targets):
    return 0


