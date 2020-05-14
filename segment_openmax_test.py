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

from evt_fitting import weibull_tailfitting

from openmax import *

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
    'num_workers': 0,             # Number of workers on data loader.
    'print_freq': 1,              # Printing frequency for mini-batch loss.
    'w_size': 224,                # Width size for image resizing.
    'h_size': 224,                # Height size for image resizing.
    'test_freq': 1200,            # Test each test_freq epochs.
    'save_freq': 1200,            # Save model each save_freq epochs.
    'open_threshold': 0.7,        # Threshold for OpenSet.
    'distance_type': 'eucos'      # Distance type for OpenMax.
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
    args['num_workers'] = 0

# Setting experiment name.
exp_name = conv_name + '_' + dataset_name + '_openmax_' + args['hidden_classes']

pretrained_path = os.path.join(ckpt_path, exp_name.replace('openmax', 'base'), 'model_' + str(epoch) + '.pth')
print('pretrained_path: "' + pretrained_path + '"')

# Setting device [0|1|2].
args['device'] = 0

# Main function.
def main(train_args):

    # Setting network architecture.
    if (conv_name == 'unet'):

        net = UNet(3, num_classes=list_dataset.num_classes, hidden_classes=hidden).cuda(args['device'])
        
    elif (conv_name == 'fcnresnet50'):

        net = FCNResNet50(3, num_classes=list_dataset.num_classes, pretrained=False, skip=True, hidden_classes=hidden).cuda(args['device'])
        
    elif (conv_name == 'fcnresnet50pretrained'):

        net = FCNResNet50(3, num_classes=list_dataset.num_classes, pretrained=True, skip=True, hidden_classes=hidden).cuda(args['device'])
        args['lr'] *= 0.1
        
    elif (conv_name == 'fcnresnext50'):

        net = FCNResNext50(3, num_classes=list_dataset.num_classes, pretrained=False, skip=True, hidden_classes=hidden).cuda(args['device'])
        
    elif (conv_name == 'fcnwideresnet50'):

        net = FCNWideResNet50(3, num_classes=list_dataset.num_classes, pretrained=False, skip=True, hidden_classes=hidden).cuda(args['device'])
        
    elif (conv_name == 'fcndensenet121'):

        net = FCNDenseNet121(3, num_classes=list_dataset.num_classes, pretrained=False, skip=True, hidden_classes=hidden).cuda(args['device'])
        
    elif (conv_name == 'fcndensenet121pretrained'):

        net = FCNDenseNet121(3, num_classes=list_dataset.num_classes, pretrained=True, skip=True, hidden_classes=hidden).cuda(args['device'])
        args['lr'] *= 0.1
        
    elif (conv_name == 'fcnvgg19'):

        net = FCNVGG19(3, num_classes=list_dataset.num_classes, pretrained=False, skip=True, hidden_classes=hidden).cuda(args['device'])
        
    elif (conv_name == 'fcnvgg19pretrained'):

        net = FCNVGG19(3, num_classes=list_dataset.num_classes, pretrained=True, skip=True, hidden_classes=hidden).cuda(args['device'])
        args['lr'] *= 0.1
        
    elif (conv_name == 'fcninceptionv3'):

        net = FCNInceptionv3(3, num_classes=list_dataset.num_classes, pretrained=False, skip=True, hidden_classes=hidden).cuda(args['device'])

    elif (conv_name == 'segnet'):

        net = SegNet(3, num_classes=list_dataset.num_classes, hidden_classes=hidden).cuda(args['device'])
        
    print('Loading pretrained weights from file "' + pretrained_path + '"')
    net.load_state_dict(torch.load(pretrained_path))
    print(net)
    
    train_args['best_record'] = {'epoch': 0, 'lr': 1e-4, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'iou': 0}

    # Setting datasets.
    val_set = list_dataset.ListDataset(dataset_name, 'Validate', (train_args['h_size'], train_args['w_size']), 'statistical', hidden)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=train_args['num_workers'], shuffle=False)

    test_set = list_dataset.ListDataset(dataset_name, 'Test', (train_args['h_size'], train_args['w_size']), 'statistical', hidden)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=train_args['num_workers'], shuffle=False)

    # Setting criterion.
    criterion = CrossEntropyLoss2d(size_average=False, ignore_index=5).cuda(args['device'])

    # Making sure checkpoint and output directories are created.
    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    check_mkdir(outp_path)
    check_mkdir(os.path.join(outp_path, exp_name))
    
    # Validation function.
    mean_list, dist_list = validate(val_loader, net, criterion, epoch, num_known_classes, num_unknown_classes, hidden, train_args)
    
    
    print('BEFORE: mean_list', sys.getsizeof(mean_list) / (1024**2), 'dist_list', sys.getsizeof(dist_list) / (1024**2))
    sys.stdout.flush()
    
    print('Fitting model...')
    tic = time.time()
    weibull_model = weibull_tailfitting(mean_list, dist_list, num_known_classes, tailsize=1000)
#     weibull_model = weibull_tailfitting(mean_list, dist_list, num_known_classes, tailsize=10000)
    toc = time.time()
    print('    Model Fitted - Elapsed Time %.2f' % (toc - tic))
    
    mean_list.clear()
    dist_list.clear()
    
    print('AFTER: mean_list', sys.getsizeof(mean_list) / (1024**2), 'dist_list', sys.getsizeof(dist_list) / (1024**2))
    
    print('weibull_model', sys.getsizeof(weibull_model) / (1024**2))
    
    sys.stdout.flush()
    
    del mean_list
    del dist_list
    
    gc.collect()
    
    # Computing test.
    test(test_loader, net, criterion, epoch, num_known_classes, num_unknown_classes, hidden, train_args, True, epoch % args['save_freq'] == 0, weibull_model)
    
    print('Exiting...')

def get_distance(arr1, arr2, distance_type):
    
    dist = 0.0
    
    if distance_type == 'euclidean':
        
        dist = np.square(arr1 - arr2)
        dist = np.sqrt(dist.sum(axis=1)) / 200.
    
    elif distance_type == 'cosine':
        
        dist = 1.0 - ((arr1 * arr2).sum(axis=1) / (np.linalg.norm(arr1, ord=2, axis=1) * np.linalg.norm(arr2, ord=2)))
    
    elif distance_type == 'eucos':
        
        dist_euc = np.square(arr1 - arr2)
        dist_euc = np.sqrt(dist_euc.sum(axis=1)) / 200.
        
        dist_cos = 1.0 - ((arr1 * arr2).sum(axis=1) / (np.linalg.norm(arr1, ord=2, axis=1) * np.linalg.norm(arr2, ord=2)))
        
        dist = dist_euc + dist_cos
    
    return dist

def update_mean_list(mean_list, outs, true, prds, num_known_classes):
    
    outs_flat = outs.permute(1, 0, 2, 3).contiguous().view(num_known_classes, -1)
    true_flat = true.view(-1)
    prds_flat = prds.view(-1)
    
    for c in range(num_known_classes):
        
        cl_outs_flat = outs_flat[:, (true_flat == c) & (prds_flat == c)].cpu().numpy()
        
        cl_mean = cl_outs_flat.mean(axis=1)
        if not np.isnan(cl_mean).any():
            
            mean_list[c].append(cl_mean)
    
    return mean_list
    
def update_dist_list(mean_list, dist_list, outs, true, prds, num_known_classes, type_distance):
    
    outs_flat = outs.permute(1, 0, 2, 3).contiguous().view(num_known_classes, -1)
    true_flat = true.view(-1)
    prds_flat = prds.view(-1)
    
    for c in range(num_known_classes):
        
        cl_outs_flat = outs_flat[:, (true_flat == c) & (prds_flat == c)].permute(1, 0).cpu().numpy()
        
        if not np.isnan(cl_outs_flat).any() and cl_outs_flat.shape[0] > 0:
            dist_list[c].append(get_distance(cl_outs_flat, mean_list[c], type_distance))
        
    return dist_list
    
def validate(val_loader, net, criterion, epoch, num_known_classes, num_unknown_classes, hidden, train_args):
    
    mean_list = [[] for c in range(num_known_classes)]
    dist_list = [[] for c in range(num_known_classes)]
    
    # Setting network for evaluation mode.
    net.eval()
    
    tic = time.time()
    
    with torch.no_grad():
        
        # Lists for whole epoch loss.
        inps_all, labs_all, prds_all, true_all = [], [], [], []

        # Creating output directory.
        check_mkdir(os.path.join(outp_path, exp_name, 'epoch_' + str(epoch)))

        #####################################################
        # Iterating over batches (computing means). #########
        #####################################################
        for i, data in enumerate(val_loader):
            
            print('Validation Mean Batch %d/%d' % (i + 1, len(val_loader)))
            sys.stdout.flush()
            
            # Obtaining images, labels and paths for batch.
            inps_batch, labs_batch, true_batch, img_name = data
            
            inps_batch = inps_batch.squeeze()
            labs_batch = labs_batch.squeeze()
            true_batch = true_batch.squeeze()
            
            # Iterating over patches inside batch.
            for j in range(inps_batch.size(0)):
                
                for k in range(inps_batch.size(1)):
                
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
                    prds = soft_outs.data.max(1)[1]

                    # Computing MAVs and distances.
                    mean_list = update_mean_list(mean_list, outs, true, prds, num_known_classes)

        for c in range(num_known_classes):
            mean_list[c] = np.asarray(mean_list[c]).mean(axis=0)
        
        #####################################################
        # Iterating over batches (computing distances). #####
        #####################################################
        for i, data in enumerate(val_loader):
            
            print('Validation Distance Batch %d/%d' % (i + 1, len(val_loader)))
            sys.stdout.flush()

            # Obtaining images, labels and paths for batch.
            inps_batch, labs_batch, true_batch, img_name = data

            inps_batch = inps_batch.squeeze()
            labs_batch = labs_batch.squeeze()
            true_batch = true_batch.squeeze()

            # Iterating over patches inside batch.
            for j in range(inps_batch.size(0)):
                
                for k in range(inps_batch.size(1)):

                    inps = inps_batch[j, k].unsqueeze(0)
                    labs = labs_batch[j, k].unsqueeze(0)
                    true = true_batch[j, k].unsqueeze(0)

                    # Casting to cuda variables.
                    inps = Variable(inps).cuda(args['device'])
                    labs = Variable(labs).cuda(args['device'])
                    true = Variable(true).cuda(args['device'])

                    # Forwarding.
                    outs = net(inps)

                    # Computing loss.
                    soft_outs = F.softmax(outs, dim=1)

                    # Obtaining predictions.
                    prds = soft_outs.data.max(1)[1]

                    # Computing MAVs and distances.
                    dist_list = update_dist_list(mean_list, dist_list, outs, true, prds, num_known_classes, args['distance_type'])
        
        for c in range(num_known_classes):
            tmp = []
            for j in range(len(dist_list[c])):
                tmp += list(dist_list[c][j])
            dist_list[c] = np.asarray(tmp)
        
    toc = time.time()
    print('    Validation finished - Elapsed Time %.2f' % (toc - tic))
    
    return mean_list, dist_list

def test(test_loader, net, criterion, epoch, num_known_classes, num_unknown_classes, hidden, train_args, save_images, save_model, weibull_model):

    # Setting network for evaluation mode.
    net.eval()
    
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
                    
                    open_outs = recalibrate_scores(weibull_model,
                                                   outs.permute(0, 2, 3, 1).cpu().numpy(),
                                                   soft_outs.permute(0, 2, 3, 1).cpu().numpy(),
                                                   num_known_classes,
                                                   alpharank=num_known_classes,
                                                   distance_type=args['distance_type'])
                    
                    
                    open_outs = np.asarray(open_outs)
                    open_outs = open_outs.reshape(outs.size(0), outs.size(2), outs.size(3), open_outs.shape[1])

                    # Obtaining predictions.
                    prds = open_outs.argmax(axis=3)
                    prds[open_outs.max(axis=3) < args['open_threshold']] = num_known_classes

                    # Appending images for epoch loss calculation.
                    inps_np = inps.detach().squeeze(0).cpu().numpy()
                    labs_np = labs.detach().squeeze(0).cpu().numpy()
                    true_np = true.detach().squeeze(0).cpu().numpy()

                    # Saving predictions.
                    if (save_images):

                        imag_path = os.path.join(outp_path, exp_name, 'epoch_' + str(epoch), img_name[0].replace('.tif', '_imag_' + str(j) + '_' + str(k) + '.png'))
                        mask_path = os.path.join(outp_path, exp_name, 'epoch_' + str(epoch), img_name[0].replace('.tif', '_mask_' + str(j) + '_' + str(k) + '.png'))
                        true_path = os.path.join(outp_path, exp_name, 'epoch_' + str(epoch), img_name[0].replace('.tif', '_true_' + str(j) + '_' + str(k) + '.png'))
                        pred_path = os.path.join(outp_path, exp_name, 'epoch_' + str(epoch), img_name[0].replace('.tif', '_pred_' + str(j) + '_' + str(k) + '.png'))
                        prob_path = os.path.join(outp_path, exp_name, 'epoch_' + str(epoch), img_name[0].replace('.tif', '_prob_' + str(j) + '_' + str(k) + '.npy'))

                        io.imsave(imag_path, np.transpose(inps_np, (1, 2, 0)))
                        io.imsave(mask_path, util.img_as_ubyte(labs_np))
                        io.imsave(true_path, util.img_as_ubyte(true_np))
                        io.imsave(pred_path, util.img_as_ubyte(prds.squeeze()))

                        np.save(prob_path, np.transpose(open_outs.squeeze(), (2, 0, 1)))
                
                toc = time.time()
                print('        Elapsed Time: %.2f' % (toc - tic))
                
        sys.stdout.flush()

if __name__ == '__main__':
    main(args)
