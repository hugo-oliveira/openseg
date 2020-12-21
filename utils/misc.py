import os
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
#             if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
    	#print(inputs.shape)
        return self.nll_loss(F.log_softmax(inputs, dim = 1), targets)

class LovaszLoss(nn.Module):

    def __init__(self, prox=False, max_steps=20):
        super(LovaszLoss, self).__init__()
        self.prox = prox
        self.max_steps=max_steps

    def forward(self, inputs, labels):
        inputs = inputs[:, 1, :, :]
        # image-level Lovasz hinge
        if inputs.size(0) == 1:
            # single image case
            loss = lovasz_single(inputs.squeeze(0), labels.squeeze(0), self.prox, self.max_steps, {})
        else:
            losses = []
            for input, label in zip(inputs, labels):
                loss = lovasz_single(input, label, self.prox, self.max_steps, {})
                losses.append(loss)
            loss = sum(losses) / len(losses)
        return loss

def naive_single(logit, label):
    # single images
    mask = (label.view(-1) != 255)
    num_preds = mask.long().sum()
#    if num_preds == 0:
#        # only void pixels, the gradients should be 0
#        return logits.sum() * 0.
    target = Variable(label.contiguous().view(-1)[mask].float())
    logit = logit.contiguous().view(-1)[mask]
    prob = F.sigmoid(logit)
    intersect = target * prob
    union = target + prob - intersect
    loss = (1. - intersect / union).sum()
    return loss

def hingeloss(logits, label):
    mask = (label.view(-1) != 255)
    num_preds = mask.long().sum()
    if num_preds == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    target = label.contiguous().view(-1)[mask]
    target = 2. * target.float() - 1.  # [target == 0] = -1
    logits = logits.contiguous().view(-1)[mask]
    hinge = 1./num_preds * F.relu(1. - logits * Variable(target)).sum()
    return hinge

def gamma_fast(gt, permutation):
    p = len(permutation)
    gt = gt.gather(0, Variable(permutation.cuda()))
    gts = gt.sum()
#    print(gts)
#    print(gt.long().cumsum(0))
    ###########################################################################
#    intersection = gts - gt.float().cumsum(0)
#    union = gts + (1 - gt).float().cumsum(0)
    intersection = gts - gt.long().cumsum(0)
    union = gts + (1 - gt).long().cumsum(0)

    jaccard = 1. - intersection / union

    jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovaszloss(logits, labels, prox=False, max_steps=20, debug={}):
    # image-level Lovasz hinge
    if logits.size(0) == 1:
        # single image case
        loss = lovasz_single(logits.squeeze(0), labels.squeeze(0), prox, max_steps, debug)
    else:
        losses = []
        for logit, label in zip(logits, labels):
            loss = lovasz_single(logit, label, prox, max_steps, debug)
            losses.append(loss)
        loss = sum(losses) / len(losses)
    return loss

def naiveloss(logits, labels):
    # image-level Lovasz hinge
    if logits.size(0) == 1:
        # single image case
        loss = naive_single(logits.squeeze(0), labels.squeeze(0))
    else:
        losses = []
        for logit, label in zip(logits, labels):
            loss = naive_single(logit, label)
            losses.append(loss)
        loss = sum(losses) / len(losses)
    return loss

def iouloss(pred, gt):
    # works for one binary pred and associated target
    # make byte tensors
    pred = (pred == 1)
    mask = (gt != 255)
    gt = (gt == 1)
    union = (gt | pred)[mask].long().sum()
    if not union:
        return 0.
    else:
        intersection = (gt & pred)[mask].long().sum()
        return 1. - intersection / union

def compute_step_length(x, grad, active, eps=1e-6):
    # compute next intersection with an edge in the direction grad
    # OR next intersection with a 0 - border
    # returns: delta in ind such that:
    # after a step delta in the direction grad, x[ind] and x[ind+1] will be equal
    delta = np.inf
    ind = -1
    if active > 0:
        numerator = (x[:active] - x[1:active+1])           # always positive (because x is sorted)
        denominator = (grad[:active] - grad[1:active+1])
        # indices corresponding to negative denominator won't intersect
        # also, we are not interested in indices in x that are *already equal*
        valid = (denominator > eps) & (numerator > eps)
        valid_indices = valid.nonzero()
        intersection_times = numerator[valid] / denominator[valid]
        if intersection_times.size():
            delta, ind = intersection_times.min(0)
            ind = valid_indices[ind]
            delta, ind = delta[0], ind[0, 0]
    if grad[active] > 0:
        intersect_zero = x[active] / grad[active]
        if intersect_zero > 0. and intersect_zero < delta:
            return intersect_zero, -1
    return delta, ind

def project(gam, active, members):
    tovisit = set(range(active + 1))
    while tovisit:
        v = tovisit.pop()
        if len(members[v]) > 1:
            avg = 0.
            for k in members[v]:
                if k != v: tovisit.remove(k)
                avg += gam[k] / len(members[v])
            for k in members[v]:
                gam[k] = avg
    if active + 1 < len(gam):
        gam[active + 1:] = 0.

def find_proximal(x0, gam, lam, eps=1e-6, max_steps=20, debug={}):
    # x0: sorted margins data
    # gam: initial gamma_fast(target, perm)
    # regularisation parameter lam
    x = x0.clone()
    act = (x >= eps).nonzero()
    finished = False
    if not act.size():
        finished = True
    else:
        active = act[-1, 0]
        members = {i: {i} for i in range(active + 1)}
        if active > 0:
            equal = (x[:active] - x[1:active+1]) < eps
            for i, e in enumerate(equal):
                if e:
                    members[i].update(members[i + 1])
                    members[i + 1] = members[i]
            project(gam, active, members)
    step = 0
    while not finished and step < max_steps and active > -1:
        step += 1
        res = compute_step_length(x, gam, active, eps)
        delta, ind = res

        if ind == -1:
            active = active - len(members[active])

        stop = torch.dot(x - x0, gam) / torch.dot(gam, gam) + 1. / lam
        if 0 <= stop < delta:
            delta = stop
            finished = True

        x = x - delta * gam
        if not finished:
            if ind >= 0:
                repr = min(members[ind])
                members[repr].update(members[ind + 1])
                for m in members[ind]:
                    if m != repr:
                        members[m] = members[repr]
            project(gam, active, members)
        if "path" in debug:
            debug["path"].append(x.numpy())

    if "step" in debug:
        debug["step"] = step
    if "finished" in debug:
        debug["finished"] = finished
    return x, gam


def lovasz_binary(margins, label, prox=False, max_steps=20, debug={}):
    # 1d vector inputs
    # Workaround: can't sort Variable bug
    # prox: False or lambda regularization value
    _, perm = torch.sort(margins.data, dim=0, descending=True)
    margins_sorted = margins[perm]
    grad = gamma_fast(label, perm)
    #loss = torch.dot(F.relu(margins_sorted), Variable(grad))
#    print(F.relu(margins_sorted))
#    print(grad)
    loss = torch.dot(F.relu(margins_sorted), grad.double().cpu())
    if prox is not False:
        xp, gam = find_proximal(margins_sorted.data, grad, prox, max_steps=max_steps, eps=1e-6, debug=debug)
        hook = margins_sorted.register_hook(lambda grad: Variable(margins_sorted.data - xp))
        return loss, hook, gam
    else:
        return loss


def lovasz_single(logit, label, prox=False, max_steps=20, debug={}):
    # single images
    #mask = (label.view(-1) != 255)
    mask = (label.view(-1) != 77)
    #print(mask.size())
#    num_preds = int(mask.long().sum())
#    print(num_preds)
#    if num_preds == 0:
        # only void pixels, the gradients should be 0
#        return logits.sum() * 0.
    target = label.contiguous().view(-1)[mask]
    signs = 2. * target.float() - 1.
    #print(logit.contiguous().view(-1).size())
    logit = logit.contiguous().view(-1)[mask]

    #margins = (1. - logit * Variable(signs))
    matr = (logit * signs).double().cpu()
    ones = Variable(torch.from_numpy(np.ones(logit.size())))
    margins = (ones - matr)
    #margins = (ones - matr).long()

    loss = lovasz_binary(margins, target, prox, max_steps, debug=debug)
    return loss

class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=True, ignore_index=-1):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs)) ** self.gamma * F.log_softmax(inputs), targets)


def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def evaluate(predictions, gts, num_classes):
    
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    if len(iu.shape) > 1:
        iu = iu[1]

    return acc, acc_cls, mean_iu, iu, fwavacc


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val_list = list()
        #self.val = 0
        #self.avg = 0
        #self.sum = 0
        #self.count = 0

    def update(self, val, n=1):
        self.val_list.append(val)
        #self.val = val
        #self.sum += val * n
        #self.count += n
        #self.avg = self.sum / self.count
        #self.std = self.sum / self.count

    def avg(self):
        return np.asarray(self.val_list).mean()

    def std(self):
        return np.asarray(self.val_list).std()

class PolyLR(object):
    def __init__(self, optimizer, curr_iter, max_iter, lr_decay):
        self.max_iter = float(max_iter)
        self.init_lr_groups = []
        for p in optimizer.param_groups:
            self.init_lr_groups.append(p['lr'])
        self.param_groups = optimizer.param_groups
        self.curr_iter = curr_iter
        self.lr_decay = lr_decay

    def step(self):
        for idx, p in enumerate(self.param_groups):
            p['lr'] = self.init_lr_groups[idx] * (1 - self.curr_iter / self.max_iter) ** self.lr_decay


# just a try, not recommend to use
class Conv2dDeformable(nn.Module):
    def __init__(self, regular_filter, cuda=True):
        super(Conv2dDeformable, self).__init__()
        assert isinstance(regular_filter, nn.Conv2d)
        self.regular_filter = regular_filter
        self.offset_filter = nn.Conv2d(regular_filter.in_channels, 2 * regular_filter.in_channels, kernel_size=3,
                                       padding=1, bias=False)
        self.offset_filter.weight.data.normal_(0, 0.0005)
        self.input_shape = None
        self.grid_w = None
        self.grid_h = None
        self.cuda = cuda

    def forward(self, x):
        x_shape = x.size()  # (b, c, h, w)
        offset = self.offset_filter(x)  # (b, 2*c, h, w)
        offset_w, offset_h = torch.split(offset, self.regular_filter.in_channels, 1)  # (b, c, h, w)
        offset_w = offset_w.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
        offset_h = offset_h.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
        if not self.input_shape or self.input_shape != x_shape:
            self.input_shape = x_shape
            grid_w, grid_h = np.meshgrid(np.linspace(-1, 1, x_shape[3]), np.linspace(-1, 1, x_shape[2]))  # (h, w)
            grid_w = torch.Tensor(grid_w)
            grid_h = torch.Tensor(grid_h)
            if self.cuda:
                grid_w = grid_w.cuda()
                grid_h = grid_h.cuda()
            self.grid_w = nn.Parameter(grid_w)
            self.grid_h = nn.Parameter(grid_h)
        offset_w = offset_w + self.grid_w  # (b*c, h, w)
        offset_h = offset_h + self.grid_h  # (b*c, h, w)
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3])).unsqueeze(1)  # (b*c, 1, h, w)
        x = F.grid_sample(x, torch.stack((offset_h, offset_w), 3))  # (b*c, h, w)
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))  # (b, c, h, w)
        x = self.regular_filter(x)
        return x


def sliced_forward(single_forward):
    def _pad(x, crop_size):
        h, w = x.size()[2:]
        pad_h = max(crop_size - h, 0)
        pad_w = max(crop_size - w, 0)
        x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, pad_h, pad_w

    def wrapper(self, x):
        batch_size, _, ori_h, ori_w = x.size()
        if self.training and self.use_aux:
            outputs_all_scales = Variable(torch.zeros((batch_size, self.num_classes, ori_h, ori_w))).cuda()
            aux_all_scales = Variable(torch.zeros((batch_size, self.num_classes, ori_h, ori_w))).cuda()
            for s in self.scales:
                new_size = (int(ori_h * s), int(ori_w * s))
                scaled_x = F.upsample(x, size=new_size, mode='bilinear')
                scaled_x = Variable(scaled_x).cuda()
                scaled_h, scaled_w = scaled_x.size()[2:]
                long_size = max(scaled_h, scaled_w)
                print(scaled_x.size())

                if long_size > self.crop_size:
                    count = torch.zeros((scaled_h, scaled_w))
                    outputs = Variable(torch.zeros((batch_size, self.num_classes, scaled_h, scaled_w))).cuda()
                    aux_outputs = Variable(torch.zeros((batch_size, self.num_classes, scaled_h, scaled_w))).cuda()
                    stride = int(ceil(self.crop_size * self.stride_rate))
                    h_step_num = int(ceil((scaled_h - self.crop_size) / stride)) + 1
                    w_step_num = int(ceil((scaled_w - self.crop_size) / stride)) + 1
                    for yy in xrange(h_step_num):
                        for xx in xrange(w_step_num):
                            sy, sx = yy * stride, xx * stride
                            ey, ex = sy + self.crop_size, sx + self.crop_size
                            x_sub = scaled_x[:, :, sy: ey, sx: ex]
                            x_sub, pad_h, pad_w = _pad(x_sub, self.crop_size)
                            print(x_sub.size())
                            outputs_sub, aux_sub = single_forward(self, x_sub)

                            if sy + self.crop_size > scaled_h:
                                outputs_sub = outputs_sub[:, :, : -pad_h, :]
                                aux_sub = aux_sub[:, :, : -pad_h, :]

                            if sx + self.crop_size > scaled_w:
                                outputs_sub = outputs_sub[:, :, :, : -pad_w]
                                aux_sub = aux_sub[:, :, :, : -pad_w]

                            outputs[:, :, sy: ey, sx: ex] = outputs_sub
                            aux_outputs[:, :, sy: ey, sx: ex] = aux_sub

                            count[sy: ey, sx: ex] += 1
                    count = Variable(count).cuda()
                    outputs = (outputs / count)
                    aux_outputs = (outputs / count)
                else:
                    scaled_x, pad_h, pad_w = _pad(scaled_x, self.crop_size)
                    outputs, aux_outputs = single_forward(self, scaled_x)
                    outputs = outputs[:, :, : -pad_h, : -pad_w]
                    aux_outputs = aux_outputs[:, :, : -pad_h, : -pad_w]
                outputs_all_scales += outputs
                aux_all_scales += aux_outputs
            return outputs_all_scales / len(self.scales), aux_all_scales
        else:
            outputs_all_scales = Variable(torch.zeros((batch_size, self.num_classes, ori_h, ori_w))).cuda()
            for s in self.scales:
                new_size = (int(ori_h * s), int(ori_w * s))
                scaled_x = F.upsample(x, size=new_size, mode='bilinear')
                scaled_h, scaled_w = scaled_x.size()[2:]
                long_size = max(scaled_h, scaled_w)

                if long_size > self.crop_size:
                    count = torch.zeros((scaled_h, scaled_w))
                    outputs = Variable(torch.zeros((batch_size, self.num_classes, scaled_h, scaled_w))).cuda()
                    stride = int(ceil(self.crop_size * self.stride_rate))
                    h_step_num = int(ceil((scaled_h - self.crop_size) / stride)) + 1
                    w_step_num = int(ceil((scaled_w - self.crop_size) / stride)) + 1
                    for yy in xrange(h_step_num):
                        for xx in xrange(w_step_num):
                            sy, sx = yy * stride, xx * stride
                            ey, ex = sy + self.crop_size, sx + self.crop_size
                            x_sub = scaled_x[:, :, sy: ey, sx: ex]
                            x_sub, pad_h, pad_w = _pad(x_sub, self.crop_size)

                            outputs_sub = single_forward(self, x_sub)

                            if sy + self.crop_size > scaled_h:
                                outputs_sub = outputs_sub[:, :, : -pad_h, :]

                            if sx + self.crop_size > scaled_w:
                                outputs_sub = outputs_sub[:, :, :, : -pad_w]

                            outputs[:, :, sy: ey, sx: ex] = outputs_sub

                            count[sy: ey, sx: ex] += 1
                    count = Variable(count).cuda()
                    outputs = (outputs / count)
                else:
                    scaled_x, pad_h, pad_w = _pad(scaled_x, self.crop_size)
                    outputs = single_forward(self, scaled_x)
                    outputs = outputs[:, :, : -pad_h, : -pad_w]
                outputs_all_scales += outputs
            return outputs_all_scales

    return wrapper

class StableBCELoss(torch.nn.modules.Module):

    def __init__(self):

        super(StableBCELoss, self).__init__()

    def forward(self, input, target):

        print(input.size())
        print(target.size())
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()
