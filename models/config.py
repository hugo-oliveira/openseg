import os

# here (https://github.com/pytorch/vision/tree/master/torchvision/models) to find the download link of pretrained models

root = '/home/hugo/FCN/pytorch-semantic-segmentation-master/pretrained'
alexnet_path = os.path.join(root, 'AlexNet', 'alexnet-owt-4df8aa71.pth')
res101_path = os.path.join(root, 'ResNet', 'resnet101-5d3b4d8f.pth')
res152_path = os.path.join(root, 'ResNet', 'resnet152-b121ed2d.pth')
#inception_v3_path = os.path.join(root, 'Inception', 'inception_v3_google-1a9a5a14.pth')
#vgg19_bn_path = os.path.join(root, 'VggNet', 'vgg19_bn-c79401a0.pth')
vgg16_path = os.path.join(root, 'VggNet', 'vgg16-397923af.pth')
#dense201_path = os.path.join(root, 'DenseNet', 'densenet201-4c113574.pth')
vgg19_bn_path = os.path.join(root, 'SegNet', 'vgg19_bn-c79401a0.pth')


'''
vgg16 trained using caffe
visit this (https://github.com/jcjohnson/pytorch-vgg) to download the converted vgg16
'''
#vgg16_caffe_path = os.path.join(root, 'VggNet', 'vgg16-caffe.pth')
vgg16_caffe_path = os.path.join(root, 'VggNet', 'vgg16-00b39a1b.pth')
