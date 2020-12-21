import datetime
import os
import random
import time
import gc
import sys
import joblib
import numpy as np

import scipy.spatial.distance as spd
from scipy import stats

from skimage import io
from skimage import util

from sklearn import metrics
from sklearn import mixture
from sklearn import decomposition
from sklearn import linear_model

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
    'open_threshold': -50,        # Threshold for OpenSet.
    'n_components': 16,           # Number of components for dimensionality reduction.
    'num_classes': 5,             # Number of classes.
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

num_known_classes = args['num_classes'] - len(hidden)
num_unknown_classes = len(hidden)

if dataset_name == 'Potsdam':
    args['epoch_num'] = 600
    args['test_freq'] = 600
    args['save_freq'] = 600
    args['num_workers'] = 0

# Setting experiment name.
exp_name = conv_name + '_' + dataset_name + '_openipca_dsm_' + args['hidden_classes']

pretrained_path = os.path.join(ckpt_path, exp_name.replace('openipca_dsm', 'base_dsm'), 'model_' + str(epoch) + '.pth')

# Setting device [0|1|2].
args['device'] = 0

# Main function.
def main(train_args):

    # Setting network architecture.
    if (conv_name == 'fcnwideresnet50'):

        net = FCNWideResNet50(4, num_classes=args['num_classes'], pretrained=False, skip=True, hidden_classes=hidden).cuda(args['device'])
        
    elif (conv_name == 'fcndensenet121'):

        net = FCNDenseNet121(4, num_classes=args['num_classes'], pretrained=False, skip=True, hidden_classes=hidden).cuda(args['device'])
        
    print('Loading pretrained weights from file "' + pretrained_path + '"')
    net.load_state_dict(torch.load(pretrained_path))
#     print(net)
    
    train_args['best_record'] = {'epoch': 0, 'lr': 1e-4, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'iou': 0}

    # Setting datasets.
    val_set = list_dataset.ListDataset(dataset_name, 'Validate', (train_args['h_size'], train_args['w_size']), 'statistical', hidden, overlap=False, use_dsm=True)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=train_args['num_workers'], shuffle=False)

    test_set = list_dataset.ListDataset(dataset_name, 'Test', (train_args['h_size'], train_args['w_size']), 'statistical', hidden, overlap=True, use_dsm=True)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=train_args['num_workers'], shuffle=False)

    # Setting criterion.
    criterion = CrossEntropyLoss2d(size_average=False, ignore_index=5).cuda(args['device'])

    # Making sure checkpoint and output directories are created.
    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    check_mkdir(outp_path)
    check_mkdir(os.path.join(outp_path, exp_name))
    
    # Validation function.
    model_full = validate(val_loader, net, criterion, epoch, num_known_classes, num_unknown_classes, hidden, train_args)
    
    # Computing test.
    test(test_loader, net, criterion, epoch, num_known_classes, num_unknown_classes, hidden, train_args, True, epoch % args['save_freq'] == 0, model_full)
    
    print('Exiting...')

def fit_quantiles(model_list, feat_np, true_np, prds_np, cl):

    # Acquiring scores for training set sample.
    scores = np.zeros_like(prds_np, dtype=np.float)
    for c in range(num_known_classes):
        feat_msk = (prds_np == c)
        if np.any(feat_msk):
            scores[feat_msk] = model_list[c].score_samples(feat_np[feat_msk, :])

    thresholds = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                  0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    scr_thresholds = np.quantile(scores, thresholds).tolist()
    
    return scr_thresholds
    
def partial_fit_ipca_model(model, feat_np, true_np, prds_np, cl):
    
    if np.any((true_np == cl) & (prds_np == cl)):
        
        cl_feat_flat = feat_np[(true_np == cl) & (prds_np == cl), :]
        
        try:
            model.partial_fit(cl_feat_flat)
        except ValueError:
            pass
        
    return model

def pred_pixelwise(model_full, feat_np, prds_np, num_known_classes, threshold):
    
    scores = np.zeros_like(prds_np, dtype=np.float)
    for c in range(num_known_classes):
        feat_msk = (prds_np == c)
        if np.any(feat_msk):
            feat_recovered = model_full['generative'][c].inverse_transform(model_full['generative'][c].transform(feat_np[feat_msk, :]))
            scores[feat_msk] = np.abs((feat_np[feat_msk, :] - feat_recovered)).sum(axis=-1)
        
    prds_np[scores < threshold] = num_known_classes
    return prds_np, scores
    
def validate(val_loader, net, criterion, epoch, num_known_classes, num_unknown_classes, hidden, train_args):
    
    model_list = []
    
    # Setting network for evaluation mode.
    net.eval()
    
    count = 0
    
    n_patches = 0
    if dataset_name == 'Vaihingen':
        n_patches = 907 # Vaihingen.
    elif dataset_name == 'Potsdam':
        n_patches = 12393 # 8993 # Potsdam.
        
    np.random.seed(12345)
    
    with torch.no_grad():
        
        ipca_training_time = [0.0 for c in range(num_known_classes)]
        
        # Creating output directory.
        check_mkdir(os.path.join(outp_path, exp_name, 'epoch_' + str(epoch)))
        
        for c in range(num_known_classes):
            
            # Computing PCA models from features.
            model = decomposition.IncrementalPCA(n_components=args['n_components'])
            
            model_list.append(model)
        
        for i, data in enumerate(val_loader):
            
            print('Validation Batch %d/%d' % (i + 1, len(val_loader)))
            sys.stdout.flush()
            
            # Obtaining images, labels and paths for batch.
            inps_batch, labs_batch, true_batch, img_name = data
            
            inps_batch = inps_batch.squeeze()
            labs_batch = labs_batch.squeeze()
            true_batch = true_batch.squeeze()
            
            # Iterating over patches inside batch.
            for j in range(inps_batch.size(0)):
                
                print('    Validation MiniBatch %d/%d' % (j + 1, inps_batch.size(0)))
                sys.stdout.flush()
                
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
                    if conv_name == 'fcnwideresnet50':
                        outs, classif1, fv2 = net(inps, feat=True)
                    elif conv_name == 'fcndensenet121':
                        outs, classif1, fv2 = net(inps, feat=True)
                    
                    # Computing loss.
                    soft_outs = F.softmax(outs, dim=1)

                    # Obtaining predictions.
                    prds = soft_outs.data.max(1)[1]
                    
                    if conv_name == 'fcnwideresnet50':
                        feat_flat = torch.cat([outs.squeeze(), classif1.squeeze(), fv2.squeeze()], 0)
                    elif conv_name == 'fcndensenet121':
                        feat_flat = torch.cat([outs.squeeze(), classif1.squeeze(), fv2.squeeze()], 0)

                    feat_flat = feat_flat.permute(1, 2, 0).contiguous().view(feat_flat.size(1) * feat_flat.size(2), feat_flat.size(0)).cpu().numpy()
                    prds_flat = prds.cpu().numpy().ravel()
                    true_flat = true.cpu().numpy().ravel()
                    
                    for c in range(num_known_classes):
                        
                        tic = time.time()
                        
                        model_list[c] = partial_fit_ipca_model(model_list[c], feat_flat, true_flat, prds_flat, c)
            
                        toc = time.time()
                        ipca_training_time[c] += (toc - tic)
            
    for c in range(num_known_classes):
        
        print('Time spent fitting model %d: %.2f' % (c, ipca_training_time[c]))
    
    model_full = {'generative': model_list}
    
    # Saving model on disk.
    model_path = os.path.join(outp_path, exp_name, 'model_pca.pkl')
    print('Saving model at "%s"...' % (model_path))
    sys.stdout.flush()
    joblib.dump(model_full, model_path)
    
    return model_full

def test(test_loader, net, criterion, epoch, num_known_classes, num_unknown_classes, hidden, train_args, save_images, save_model, model_full):

    # Setting network for evaluation mode.
    net.eval()
    
    with torch.no_grad():

        # Creating output directory.
        check_mkdir(os.path.join(outp_path, exp_name, 'epoch_' + str(epoch)))

        # Iterating over batches.
        for i, data in enumerate(test_loader):
            
            print('Test Batch %d/%d' % (i + 1, len(test_loader)))
            sys.stdout.flush()

            # Obtaining images, labels and paths for batch.
            inps_batch, labs_batch, true_batch, img_name = data

            inps_batch = inps_batch.squeeze()
            labs_batch = labs_batch.squeeze()
            true_batch = true_batch.squeeze()

            # Iterating over patches inside batch.
            for j in range(inps_batch.size(0)):
            
                print('    Test MiniBatch %d/%d' % (j + 1, inps_batch.size(0)))
                sys.stdout.flush()
                
                tic = time.time()
                
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
                    if conv_name == 'fcnwideresnet50':
                        outs, classif1, fv2 = net(inps, feat=True)
                    elif conv_name == 'fcndensenet121':
                        outs, classif1, fv2 = net(inps, feat=True)

                    # Computing probabilities.
                    soft_outs = F.softmax(outs, dim=1)

                    # Obtaining prior predictions.
                    prds = soft_outs.data.max(1)[1]

                    # Obtaining posterior predictions.
                    if conv_name == 'fcnwideresnet50':
                        feat_flat = torch.cat([outs, classif1, fv2], 1)
                    elif conv_name == 'fcndensenet121':
                        feat_flat = torch.cat([outs, classif1, fv2], 1)
                        
                    feat_flat = feat_flat.permute(0, 2, 3, 1).contiguous().view(feat_flat.size(0) * feat_flat.size(2) * feat_flat.size(3), feat_flat.size(1)).cpu().numpy()
                    prds_flat = prds.cpu().numpy().ravel()
                    true_flat = true.cpu().numpy().ravel()

                    prds_post, scores = pred_pixelwise(model_full, feat_flat, prds_flat, num_known_classes, 0.0)
                    prds_post = prds_post.reshape(prds.size(0), prds.size(1), prds.size(2))
                    scores = scores.reshape(prds.size(0), prds.size(1), prds.size(2))

                    # Saving predictions.
                    if (save_images):

                        pred_prev_path = os.path.join(outp_path, exp_name, 'epoch_' + str(epoch), img_name[0].replace('.tif', '_prev_' + str(j) + '_' + str(k) + '.png'))
                        scor_path = os.path.join(outp_path, exp_name, 'epoch_' + str(epoch), img_name[0].replace('.tif', '_scor_' + str(j) + '_' + str(k) + '.npy'))
                        
                        io.imsave(pred_prev_path, util.img_as_ubyte(prds.cpu().squeeze().numpy()))
                        np.save(scor_path, scores.squeeze())
                
                toc = time.time()
                print('        Elapsed Time: %.2f' % (toc - tic)) 
                
        sys.stdout.flush()

if __name__ == '__main__':
    main(args)
