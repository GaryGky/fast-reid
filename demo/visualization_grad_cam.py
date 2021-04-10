import sys

import torch

sys.path.append('.')

from torchvision import transforms
from fastreid.config import cfg
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

from fastreid.modeling.backbones.resnet import build_resnet_backbone

if __name__ == '__main__':
    cfg.MODEL.BACKBONE.NAME = 'resnet18'
    cfg.MODEL.BACKBONE.DEPTH = "18x"
    cfg.MODEL.BACKBONE.WITH_IBN = True
    cfg.MODEL.BACKBONE.WITH_SE = True
    cfg.MODEL.BACKBONE.PRETRAIN_PATH = 'logs/veri/sbs_R50-ibn/model_best.pth'

    model = build_resnet_backbone(cfg)
    # print(model.eval())

    image = Image.open('demo/img/orange.png')

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    img = transform(image)
    img = img.unsqueeze(0)

    # forward pass
    pred = model(img)
    # model.forward(img)
