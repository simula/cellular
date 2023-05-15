import torch
import torchvision.models as models

import segmentation_models_pytorch as smp

def build_model(name, **params):
    return __MODELS[name](**params)

def smp_model(model_cls, encoder_name, encoder_weights, *argv, **kwargs):
    model = model_cls(encoder_name=encoder_name, encoder_weights=encoder_weights,
                      *argv, **kwargs)
    return model

def torch_model(model_cls, out_features, classes=None, *argv, **kwargs):
    model = model_cls(*argv, **kwargs)
    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features, out_features=out_features, bias=True)
    return model

__MODELS = {
    "resnet18": lambda *argv, **kwargs:  torch_model(models.resnet18, *argv, **kwargs),
    "alexnet": lambda *argv, **kwargs:  torch_model(models.alexnet, *argv, **kwargs),
    "vgg16": lambda *argv, **kwargs:  torch_model(models.vgg16, *argv, **kwargs),
    "squeezenet": lambda *argv, **kwargs:  torch_model(models.squeezenet1_0, *argv, **kwargs),
    "densenet": lambda *argv, **kwargs:  torch_model(models.densenet161, *argv, **kwargs),
    "inception": lambda *argv, **kwargs:  torch_model(models.inception_v3, *argv, **kwargs),
    "googlenet": lambda *argv, **kwargs:  torch_model(models.googlenet, *argv, **kwargs),
    "shufflenet": lambda *argv, **kwargs:  torch_model(models.shufflenet_v2_x1_0, *argv, **kwargs),
    "mobilenet_v2": lambda *argv, **kwargs:  torch_model(models.mobilenet_v2, *argv, **kwargs),
    "mobilenet_v3_large": lambda *argv, **kwargs:  torch_model(models.mobilenet_v3_large, *argv, **kwargs),
    "mobilenet_v3_small": lambda *argv, **kwargs:  torch_model(models.mobilenet_v3_small, *argv, **kwargs),
    "resnext50_32x4d": lambda *argv, **kwargs:  torch_model(models.resnext50_32x4d, *argv, **kwargs),
    "wide_resnet50_2": lambda *argv, **kwargs:  torch_model(models.wide_resnet50_2, *argv, **kwargs),
    "mnasnet": lambda *argv, **kwargs:  torch_model(models.mnasnet1_0, *argv, **kwargs),
    "efficientnet_b0": lambda *argv, **kwargs:  torch_model(models.efficientnet_b0, *argv, **kwargs),
    "efficientnet_b1": lambda *argv, **kwargs:  torch_model(models.efficientnet_b1, *argv, **kwargs),
    "efficientnet_b2": lambda *argv, **kwargs:  torch_model(models.efficientnet_b2, *argv, **kwargs),
    "efficientnet_b3": lambda *argv, **kwargs:  torch_model(models.efficientnet_b3, *argv, **kwargs),
    "efficientnet_b4": lambda *argv, **kwargs:  torch_model(models.efficientnet_b4, *argv, **kwargs),
    "efficientnet_b5": lambda *argv, **kwargs:  torch_model(models.efficientnet_b5, *argv, **kwargs),
    "efficientnet_b6": lambda *argv, **kwargs:  torch_model(models.efficientnet_b6, *argv, **kwargs),
    "efficientnet_b7": lambda *argv, **kwargs:  torch_model(models.efficientnet_b7, *argv, **kwargs),

    "smp_Unet": lambda *argv, **kwargs: smp_model(smp.Unet, *argv, **kwargs),
    "smp_UnetPlusPlus": lambda *argv, **kwargs: smp_model(smp.UnetPlusPlus, *argv, **kwargs),
    "smp_MAnet": lambda *argv, **kwargs: smp_model(smp.MAnet, *argv, **kwargs),
    "smp_Linknet": lambda *argv, **kwargs: smp_model(smp.Linknet, *argv, **kwargs),
    "smp_FPN": lambda *argv, **kwargs: smp_model(smp.FPN, *argv, **kwargs),
    "smp_PSPNet": lambda *argv, **kwargs: smp_model(smp.PSPNet, *argv, **kwargs),
    "smp_PAN": lambda *argv, **kwargs: smp_model(smp.PAN, *argv, **kwargs),
    "smp_DeepLabV3": lambda *argv, **kwargs: smp_model(smp.DeepLabV3, *argv, **kwargs),
    "smp_DeepLabV3Plus": lambda *argv, **kwargs: smp_model(smp.DeepLabV3Plus, *argv, **kwargs),
}
    
    
    
    
    
    
    
    
    