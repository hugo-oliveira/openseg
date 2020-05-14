import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

from utils import get_upsampling_weight
from .config import vgg16_caffe_path

from utils import initialize_weights

class FCNVGG19(nn.Module):
    
    def __init__(self, input_channels, num_classes, pretrained=True, skip=True, hidden_classes=None):

        super(FCNVGG19, self).__init__()

        self.skip = skip
        
        # VGG-19 BN with Skip Connections (adapted from FCN-8s).
        vgg19bn = models.vgg19_bn(pretrained=pretrained, progress=False)

        if pretrained:
            self.block1 = vgg19bn.features[:7]
        else:
            self.block1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                *vgg19bn.features[1:7],
            )
        self.block2 = vgg19bn.features[7:14]
        self.block3 = vgg19bn.features[14:27]
        self.block4 = vgg19bn.features[27:40]
        self.block5 = vgg19bn.features[40:]
        
        if self.skip:
            
            if hidden_classes is None:
                self.classifier1 = nn.Sequential(
                    nn.Conv2d(512 + 128, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(0.5),
                )
                self.final = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
            else:
                self.classifier1 = nn.Sequential(
                    nn.Conv2d(512 + 128, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(0.5),
                )
                self.final = nn.Conv2d(64, num_classes - len(hidden_classes), kernel_size=3, padding=1)
                
        else:

            if hidden_classes is None:
                self.classifier1 = nn.Sequential(
                    nn.Conv2d(512, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(0.5),
                )
                self.final = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
            else:
                self.classifier1 = nn.Sequential(
                    nn.Conv2d(512, 64, kernel_size=3, padding=1),
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
            fv1 = self.block1(x)
            fv2 = self.block2(fv1)
            fv3 = self.block3(fv2)
            fv4 = self.block4(fv3)
            fv5 = self.block5(fv4)

            fv_final = torch.cat([F.upsample(fv2, x.size()[2:], mode='bilinear'),
                                  F.upsample(fv5, x.size()[2:], mode='bilinear')], 1)

        else:

            # Forward on FCN without Skip Connections.
            fv1 = self.block1(x)
            fv2 = self.block2(fv1)
            fv3 = self.block3(fv2)
            fv4 = self.block4(fv3)
            fv5 = self.block5(fv4)

            fv_final = F.upsample(fv5, x.size()[2:], mode='bilinear')

        classif1 = self.classifier1(fv_final)
        output = self.final(classif1)
        
        if feat:
            return (output, classif1, F.upsample(fv3, x.size()[2:], mode='bilinear'))

        else:
            return output
