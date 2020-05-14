import datetime
import os
import random
import time
import gc
import sys
import numpy as np

import scipy.spatial.distance as spd

from skimage import io
from skimage import util

from sklearn import metrics

from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.nn.functional as F

import list_dataset
from models import *
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d, LovaszLoss, FocalLoss2d

cudnn.benchmark = True

'''
Classes:
    0 = Street
    1 = Building
    2 = Grass
    3 = Tree
    4 = Car
    5 = Surfaces
    6 = Boundaries
'''

# Predefining directories.
ckpt_path = './ckpt'
imag_path = './images'
outp_path = './outputs'

# Setting predefined arguments.
args = {
    'epoch_num': 1200,            # Number of epochs.
    'lr': 1e-4,                   # Learning rate.
    'weight_decay': 5e-4,         # L2 penalty.
    'momentum': 0.9,              # Momentum.
    'num_workers': 4,             # Number of workers on data loader.
    'print_freq': 1,              # Printing frequency for mini-batch loss.
    'w_size': 224,                # Width size for image resizing.
    'h_size': 224,                # Height size for image resizing.
    'test_freq': 1200,            # Test each test_freq epochs.
    'save_freq': 1200,            # Save model each save_freq epochs.
    'open_threshold': 0.7,        # Threshold for OpenSet.
}

# Reading system parameters.
conv_name = sys.argv[1]
args['hidden_classes'] = sys.argv[2]
print('hidden: ' + sys.argv[2])

dataset_name = sys.argv[3]

epoch = int(sys.argv[4])

hidden = []
if '_' in args['hidden_classes']:
    hidden = [int(h) for h in args['hidden_classes'].split('_')]
else:
    hidden = [int(args['hidden_classes'])]

num_known_classes = list_dataset.num_classes - len(hidden)
num_unknown_classes = len(hidden)

if dataset_name == 'Potsdam':
    args['epoch_num'] = 600
    args['test_freq'] = 600
    args['save_freq'] = 600
    args['num_workers'] = 2

# Setting experiment name.
exp_name = conv_name + '_' + dataset_name + '_softmax_' + args['hidden_classes']

pretrained_path = os.path.join(ckpt_path, exp_name.replace('softmax', 'base'), 'model_' + str(epoch) + '.pth')

# Setting device [0|1|2].
args['device'] = 0

# Main function.
def main(train_args):

    # Setting network architecture.
    if (conv_name == 'unet'):

        net = UNet(3, num_classes=list_dataset.num_classes, hidden_classes=hidden).cuda(args['device'])
        
    elif (conv_name == 'fcnresnet50'):

        net = FCNResNet50(3, num_classes=list_dataset.num_classes, pretrained=False, skip=True, hidden_classes=hidden).cuda(args['device'])
        
    elif (conv_name == 'fcnresnext50'):

        net = FCNResNext50(3, num_classes=list_dataset.num_classes, pretrained=False, skip=True, hidden_classes=hidden).cuda(args['device'])
        
    elif (conv_name == 'fcnwideresnet50'):

        net = FCNWideResNet50(3, num_classes=list_dataset.num_classes, pretrained=False, skip=True, hidden_classes=hidden).cuda(args['device'])
        
    elif (conv_name == 'fcndensenet121'):

        net = FCNDenseNet121(3, num_classes=list_dataset.num_classes, pretrained=False, skip=True, hidden_classes=hidden).cuda(args['device'])
        
    elif (conv_name == 'fcninceptionv3'):

        net = FCNInceptionv3(3, num_classes=list_dataset.num_classes, pretrained=False, skip=True, hidden_classes=hidden).cuda(args['device'])
        
    elif (conv_name == 'fcnvgg19'):

        net = FCNVGG19(3, num_classes=list_dataset.num_classes, pretrained=False, skip=True, hidden_classes=hidden).cuda(args['device'])

    elif (conv_name == 'segnet'):

        net = SegNet(3, num_classes=list_dataset.num_classes, hidden_classes=hidden).cuda(args['device'])
        
    print('Loading pretrained weights from file "' + pretrained_path + '"')
    net.load_state_dict(torch.load(pretrained_path))
    print(net)
    
    train_args['best_record'] = {'epoch': 0, 'lr': 1e-4, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'iou': 0}

    # Setting datasets.
    test_set = list_dataset.ListDataset(dataset_name, 'Test', (train_args['h_size'], train_args['w_size']), 'statistical', hidden)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=train_args['num_workers'], shuffle=False)

    # Setting criterion.
    criterion = CrossEntropyLoss2d(size_average=False, ignore_index=5).cuda(args['device'])

    # Making sure checkpoint and output directories are created.
    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    check_mkdir(outp_path)
    check_mkdir(os.path.join(outp_path, exp_name))
    
    # Computing test.
    test(test_loader, net, criterion, epoch, num_known_classes, num_unknown_classes, hidden, train_args, True, epoch % args['save_freq'] == 0)
    
    print('Exiting...')

def test(test_loader, net, criterion, epoch, num_known_classes, num_unknown_classes, hidden, train_args, save_images, save_model):

    # Setting network for evaluation mode.
    net.eval()
    
    gc.collect()
    
    with torch.no_grad():

        # Creating output directory.
        check_mkdir(os.path.join(outp_path, exp_name, 'epoch_' + str(epoch)))

        # Iterating over batches.
        for i, data in enumerate(test_loader):
            
            print('Test Batch %d/%d' % (i + 1, len(test_loader)))

            # Obtaining images, labels and paths for batch.
            inps_batch, labs_batch, true_batch, img_name = data

            inps_batch = inps_batch.squeeze()
            labs_batch = labs_batch.squeeze()
            true_batch = true_batch.squeeze()

            # Iterating over patches inside batch.
            for j in range(inps_batch.size(0)):
                
                print('    Test MiniBatch %d/%d' % (j + 1, inps_batch.size(0)))
                
                tic = time.time()
                
                for k in range(inps_batch.size(1)):
                    
                    print('        Test MiniMiniBatch %d/%d' % (k + 1, inps_batch.size(1)))
            
                    sys.stdout.flush()

                    inps = inps_batch[j, k].unsqueeze(0)
                    labs = labs_batch[j, k].unsqueeze(0)
                    true = true_batch[j, k].unsqueeze(0)

                    # Casting tensors to cuda.
                    inps, labs, true = inps.cuda(args['device']), labs.cuda(args['device']), true.cuda(args['device'])

                    # Casting to cuda variables.
                    inps = Variable(inps).cuda(args['device'])
                    labs = Variable(labs).cuda(args['device'])
                    true = Variable(true).cuda(args['device'])

                    # Forwarding.
                    outs = net(inps)

                    # Computing loss.
                    soft_outs = F.softmax(outs, dim=1)

                    # Obtaining predictions.
                    prds = soft_outs.max(dim=1)[1]
                    prds[soft_outs.max(dim=1)[0] < args['open_threshold']] = num_known_classes

                    # Appending images for epoch loss calculation.
                    inps_np = inps.detach().squeeze(0).cpu().numpy()
                    labs_np = labs.detach().squeeze(0).cpu().numpy()
                    true_np = true.detach().squeeze(0).cpu().numpy()
                    
                    prds = prds.detach().squeeze(0).cpu().numpy()

                    # Saving predictions.
                    if (save_images):
                        
                        pred_path = os.path.join(outp_path, exp_name, 'epoch_' + str(epoch), img_name[0].replace('.tif', '_pred_' + str(j) + '_' + str(k) + '.png'))
                        prob_path = os.path.join(outp_path, exp_name, 'epoch_' + str(epoch), img_name[0].replace('.tif', '_prob_' + str(j) + '_' + str(k) + '.npy'))

                        io.imsave(pred_path, util.img_as_ubyte(prds))
                        np.save(prob_path, np.transpose(soft_outs.detach().cpu().numpy().squeeze(), (1, 2, 0)))
                
                toc = time.time()
                print('        Elapsed Time: %.2f' % (toc - tic))
                
        sys.stdout.flush()

if __name__ == '__main__':
    main(args)