import torch
import torch.nn as nn
from libs.models.DAN.resnet import resnet50


class Backbone(nn.Module):
    def __init__(self, backbone='resnet50'):
        super(Backbone, self).__init__()
        self.backbone = backbone
        
        # Load pretrained ResNet50
        if self.backbone == 'resnet50':
            backbone = resnet50(pretrained=True)
            # Get features from different layers
            self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu1,
                                      backbone.conv2, backbone.bn2, backbone.relu2,
                                      backbone.conv3, backbone.bn3, backbone.relu3,
                                      backbone.maxpool)
            self.layer1 = backbone.layer1  # 1/4
            self.layer2 = backbone.layer2  # 1/8
            self.layer3 = backbone.layer3  # 1/16
            self.layer4 = backbone.layer4  # 1/32
            
            # Feature dimensions for each layer
            self.feat_dims = {
                'layer1': 256,   # layer1 output
                'layer2': 512,   # layer2 output
                'layer3': 1024,  # layer3 output 
                'layer4': 2048   # layer4 output
            }
            
            self.feat_dim = 256  # Keep main feature dimension constant
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
    
    def forward(self, x):
        x = self.layer0(x)
        l1 = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(x)
        l4 = self.layer4(x)
        return l1, l2, l3, l4  # 1/4, 1/8, 1/16, 1/32
