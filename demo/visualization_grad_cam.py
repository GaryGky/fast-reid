import sys

sys.path.append('.')
from fastreid.modeling.backbones import build_resnest_backbone

from torchvision import transforms
from fastreid.config import cfg
from PIL import Image

if __name__ == '__main__':
    cfg.MODEL.BACKBONE.NAME = 'resnest'
    cfg.MODEL.BACKBONE.DEPTH = "50x"
    cfg.MODEL.BACKBONE.WITH_IBN = True
    cfg.MODEL.BACKBONE.WITH_SE = True
    cfg.MODEL.BACKBONE.PRETRAIN_PATH = 'logs/veri/resnest/sbs_Rs50-ibn/model_best.pth'

    model = build_resnest_backbone(cfg)
    model.eval()
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
