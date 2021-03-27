import sys

import cv2

sys.path.append('.')

import torch
from torchvision import transforms
import numpy as np
from fastreid.config import cfg
from PIL import Image
import matplotlib.pyplot as plt

from fastreid.modeling.backbones.resnet import build_resnet_backbone

if __name__ == '__main__':
    cfg.MODEL.BACKBONE.NAME = 'resnet18'
    cfg.MODEL.BACKBONE.DEPTH = "18x"
    cfg.MODEL.BACKBONE.WITH_IBN = True
    cfg.MODEL.BACKBONE.WITH_SE = True
    cfg.MODEL.BACKBONE.PRETRAIN_PATH = 'logs/veri/sbs_R50-ibn/model_best.pth'

    model = build_resnet_backbone(cfg)
    # print(model.eval())

    image = Image.open('demo/img/black.png')

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    img = transform(image)
    img = img.unsqueeze(0)

    # forward pass
    pred = model(img)
    #
    # pred.argmax(dim=1)  # prints tensor([2])
    #
    # # get the gradient of the output with respect to the parameters of the model
    # pred[:, 2].sum().backward()
    #
    # # pull the gradients out of the model
    # gradients = model.get_gradient()
    #
    # # pool the gradients across the channels
    # pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    #
    # # get the activations of the last convolutional layer
    # activations = model.get_activations(img).detach()
    #
    # # weight the channels by corresponding gradients
    # for i in range(512):
    #     activations[:, i, :, :] *= pooled_gradients[i]
    #
    # # average the channels of the activations
    # heatmap = torch.mean(activations, dim=1).squeeze()
    #
    # # relu on top of the heatmap
    # # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    # heatmap = np.maximum(heatmap, 0)
    #
    # # normalize the heatmap
    # heatmap /= torch.max(heatmap)
    #
    # # draw the heatmap
    # plt.matshow(heatmap.squeeze())
    #
    # # make the heatmap to be a numpy array
    # heatmap = heatmap.numpy()
    #
    # # interpolate the heatmap
    # img = cv2.imread('demo/img/black.png')
    # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # heatmap = np.uint8(255 * heatmap)
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # superimposed_img = heatmap * 0.4 + img
    # cv2.imwrite('./map.jpg', superimposed_img)
