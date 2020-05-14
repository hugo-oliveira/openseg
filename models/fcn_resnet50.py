import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

from utils import get_upsampling_weight
from .config import vgg16_caffe_path

from utils import initialize_weights

class FCNResNet50(nn.Module):
    
    def __init__(self, input_channels, num_classes, pretrained=True, skip=True, hidden_classes=None):

        super(FCNResNet50, self).__init__()

        self.skip = skip
        
        # ResNet-50 with Skip Connections (adapted from FCN-8s).
        resnet = models.resnet50(pretrained=pretrained, progress=False)

        if pretrained:
            self.init = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool
            )
        else:
            self.init = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                resnet.bn1,
                resnet.relu,
                resnet.maxpool
            )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        if self.skip:
            
            if hidden_classes is None:
                self.classifier1 = nn.Sequential(
                    nn.Conv2d(2048 + 256, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(0.5),
                )
                self.final = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
            else:
                self.classifier1 = nn.Sequential(
                    nn.Conv2d(2048 + 256, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(0.5),
                )
                self.final = nn.Conv2d(64, num_classes - len(hidden_classes), kernel_size=3, padding=1)
                
        else:

            if hidden_classes is None:
                self.classifier1 = nn.Sequential(
                    nn.Conv2d(2048, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(0.5),
                )
                self.final = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
            else:
                self.classifier1 = nn.Sequential(
                    nn.Conv2d(2048, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(0.5),
                )
                self.final = nn.Conv2d(64, num_classes - len(hidden_classes), kernel_size=3, padding=1)
        
        if not pretrained:
            initialize_weights(self)
        else:
            initialize_weights(self.classifier1)
            initialize_weights(self.final)

    def forward(self, x, feat=False):
        
        if self.skip:
            
            # Forward on FCN with Skip Connections.
            fv_init = self.init(x)
            fv1 = self.layer1(fv_init)
            fv2 = self.layer2(fv1)
            fv3 = self.layer3(fv2)
            fv4 = self.layer4(fv3)

            fv_final = torch.cat([F.upsample(fv1, x.size()[2:], mode='bilinear'),
                                  F.upsample(fv4, x.size()[2:], mode='bilinear')], 1)
#             print('fv_final.size()', fv_final.size())

        else:

            # Forward on FCN without Skip Connections.
            fv_init = self.init(x)
            fv1 = self.layer1(fv_init)
            fv2 = self.layer2(fv1)
            fv3 = self.layer3(fv2)
            fv4 = self.layer4(fv3)

            fv_final = F.upsample(fv4, x.size()[2:], mode='bilinear')
            
        classif1 = self.classifier1(fv_final)
#         print('classif1.size()', classif1.size())
        output = self.final(classif1)
        
        if feat:
            return (output, classif1, F.upsample(fv2, x.size()[2:], mode='bilinear'))

        else:
            return output
