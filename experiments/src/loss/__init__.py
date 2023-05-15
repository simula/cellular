import torch
import torch.nn as nn

import segmentation_models_pytorch as smp

def build_criterion( name, params={}, *argv, **kwargs ):
    return __CRITERION[ name ]( *argv, **params, **kwargs )

__CRITERION = {
    "DiceLoss": smp.losses.DiceLoss,
    "JaccardLoss": smp.losses.JaccardLoss,
    "FocalLoss": smp.losses.FocalLoss,

    "L1Loss": nn.L1Loss,
    "MSELoss": nn.MSELoss,
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "CTCLoss": nn.CTCLoss,
    "NLLLoss": nn.NLLLoss,
    "PoissonNLLLoss": nn.PoissonNLLLoss,
    "GaussianNLLLoss": nn.GaussianNLLLoss,
    "KLDivLoss": nn.KLDivLoss,
    "BCELoss": nn.BCELoss,
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
    "MarginRankingLoss": nn.MarginRankingLoss,
    "HingeEmbeddingLoss": nn.HingeEmbeddingLoss,
    "MultiLabelMarginLoss": nn.MultiLabelMarginLoss,
    "HuberLoss": nn.HuberLoss,
    "SmoothL1Loss": nn.SmoothL1Loss,
    "SoftMarginLoss": nn.SoftMarginLoss,
    "MultiLabelSoftMarginLoss": nn.MultiLabelSoftMarginLoss,
    "CosineEmbeddingLoss": nn.CosineEmbeddingLoss,
    "MultiMarginLoss": nn.MultiMarginLoss,
    "TripletMarginLoss": nn.TripletMarginLoss,
    "TripletMarginWithDistanceLoss": nn.TripletMarginWithDistanceLoss
}