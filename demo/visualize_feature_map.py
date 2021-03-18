import sys

from torch import nn
from torchvision import models, transforms
from torchvision.utils import make_grid
import numpy as np
from fastreid.config import cfg
from PIL import Image

sys.path.append('..')
from fastreid.modeling.backbones.resnet import build_resnet_backbone

cfg.MODEL.BACKBONE.NAME = 'resnet18'
cfg.MODEL.BACKBONE.DEPTH = 18
cfg.MODEL.BACKBONE.WITH_IBN = True
cfg.MODEL.BACKBONE.WITH_SE = True
cfg.MODEL.BACKBONE.PRETRAIN_PATH = 'logs/veri/sbs_R50-ibn/model_best.pth'

net1 = build_resnet_backbone(cfg)

image = Image.open('./img/0002_c002_00030600_0.jpg')

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
img = transform(image)
img = img.unsqueeze(0)


def save_img(tensor, name):
    tensor = tensor.permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(name + '.jpg')


new_model = nn.Sequential(*list(model.modules())[:5])
f3 = new_model(img)
save_img(f3, 'layer1')

new_model = nn.Sequential(*list(model.children())[:6])
f4 = new_model(img)  # [1, 128, 28, 28]
save_img(f4, 'layer2')

new_model = nn.Sequential(*list(model.children())[:7])
print(new_model)
f5 = new_model(img)  # [1, 256, 14, 14]
print(f5.shape)
save_img(f5, 'layer3')

new_model = nn.Sequential(*list(model.children())[:8])
print(new_model)
f6 = new_model(img)  # [1, 256, 14, 14]
print(f6.shape)
save_img(f6, 'layer4')
