import torch
import torch.nn as nn

import segmentation_models_pytorch as smp

def smp_metric(metric_func, y_pred, y_true, params):
    tp, fp, fn, tn = smp.metrics.get_stats(y_pred, y_true, mode=params.stats.mode, threshold=params.stats.threshold)
    return metric_func(tp, fp, fn, tn, **params.metric)

def call_metric( name, y_pred, y_true, params={}, *argv, **kwargs ):
    return METRIC[ name ]( y_pred, y_true, params )

METRIC = {
    "fbeta_score": lambda y_pred, y_true, params: smp_metric(smp.metrics.fbeta_score, y_pred, y_true, params),
    "f1_score": lambda y_pred, y_true, params: smp_metric(smp.metrics.f1_score, y_pred, y_true, params),
    "iou_score": lambda y_pred, y_true, params: smp_metric(smp.metrics.iou_score, y_pred, y_true, params),
    "accuracy": lambda y_pred, y_true, params: smp_metric(smp.metrics.accuracy, y_pred, y_true, params),
    "precision": lambda y_pred, y_true, params: smp_metric(smp.metrics.precision, y_pred, y_true, params),
    "recall": lambda y_pred, y_true, params: smp_metric(smp.metrics.recall, y_pred, y_true, params),
    "sensitivity": lambda y_pred, y_true, params: smp_metric(smp.metrics.sensitivity, y_pred, y_true, params),
    "specificity": lambda y_pred, y_true, params: smp_metric(smp.metrics.specificity, y_pred, y_true, params),
    "balanced_accuracy": lambda y_pred, y_true, params: smp_metric(smp.metrics.balanced_accuracy, y_pred, y_true, params),
    "positive_predictive_value": lambda y_pred, y_true, params: smp_metric(smp.metrics.positive_predictive_value, y_pred, y_true, params),
    "negative_predictive_value": lambda y_pred, y_true, params: smp_metric(smp.metrics.negative_predictive_value, y_pred, y_true, params),
    "false_negative_rate": lambda y_pred, y_true, params: smp_metric(smp.metrics.false_negative_rate, y_pred, y_true, params),
    "false_positive_rate": lambda y_pred, y_true, params: smp_metric(smp.metrics.false_positive_rate, y_pred, y_true, params),
    "false_discovery_rate": lambda y_pred, y_true, params: smp_metric(smp.metrics.false_discovery_rate, y_pred, y_true, params),
    "false_omission_rate": lambda y_pred, y_true, params: smp_metric(smp.metrics.false_omission_rate, y_pred, y_true, params),
    "positive_likelihood_ratio": lambda y_pred, y_true, params: smp_metric(smp.metrics.positive_likelihood_ratio, y_pred, y_true, params),
    "negative_likelihood_ratio": lambda y_pred, y_true, params: smp_metric(smp.metrics.negative_likelihood_ratio, y_pred, y_true, params),
}