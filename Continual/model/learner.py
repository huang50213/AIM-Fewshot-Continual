import logging
import matplotlib.pyplot as plt
from numpy.random import default_rng

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .aim import AIM

TOPK = 10

def batchnorm(input, weight=None, bias=None, running_mean=None, running_var=None, training=True, eps=1e-5, momentum=0.1):
    ''' momentum = 1 restricts stats to the current mini-batch '''
    # This hack only works when momentum is 1 and avoids needing to track running stats
    # by substuting dummy variables
    running_mean = torch.zeros(np.prod(np.array(input.data.size()[1])))
    running_var = torch.ones(np.prod(np.array(input.data.size()[1])))
    return F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)

def maxpool(input, kernel_size, stride=None):
    return F.max_pool2d(input, kernel_size, stride)

def avgpool(input, kernel_size, stride=None):
    return F.avg_pool2d(input, kernel_size, stride)

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

class Learner(nn.Module):

    def __init__(self, config, treatment):
        """
        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()

        self.config = config
        self.treatment = treatment
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()
        self.aim = None
        for i, (name, param) in enumerate(self.config):
            if 'conv' in name:
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif 'linear' in name or 'nm_to' in name:
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == 'fc':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name == 'aim':
                self.aim = AIM(param[0], param[1], param[2], TOPK, param[3], param[4], param[5], 1)
                self.vars.extend(self.aim.create_parameters())
            elif name is 'cat':
                pass
            elif name is 'cat_start':
                pass
            elif name is "rep":
                pass
            elif 'bn' in name:
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])            

            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError

    def extra_repr(self):
        info = ''
        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'
            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'
            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)' % (param[1], param[0])
                info += tmp + '\n'
            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)' % (param[0])
                info += tmp + '\n'
            elif name is 'cat':
                tmp = 'cat'
                info += tmp + "\n"
            elif name is 'cat_start':
                tmp = 'cat_start'
                info += tmp + "\n"
            elif name is 'rep':
                tmp = 'rep'
                info += tmp + "\n"
            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info

    def forward(self, x, vars=None, bn_training=True, meta_train=False, iterations=None, return_aim=False):
        if x.size(-1) == 28:
            dataset = 'omniglot'
        elif x.size(-1) == 32:
            dataset = 'cifar100'
        elif x.size(-1) == 84:
            dataset = 'imagenet'
        else:
            raise NotImplementedError
        if vars is None:
            vars = self.vars
        if self.treatment == 'ANML+AIM':
            # =========== NEUROMODULATORY NETWORK ===========
            data = x
            nm_data = x
            w,b = vars[0], vars[1]
            nm_data = conv2d(nm_data, w, b, 1)
            w,b = vars[2], vars[3]
            running_mean, running_var = self.vars_bn[0], self.vars_bn[1]
            nm_data = F.batch_norm(nm_data, running_mean, running_var, weight=w, bias=b, training=bn_training)
            nm_data = F.relu(nm_data)
            nm_data = maxpool(nm_data, kernel_size=2, stride=2)
            w,b = vars[4], vars[5]
            nm_data = conv2d(nm_data, w, b, 1)
            w,b = vars[6], vars[7]
            running_mean, running_var = self.vars_bn[2], self.vars_bn[3]
            nm_data = F.batch_norm(nm_data, running_mean, running_var, weight=w, bias=b, training=bn_training)
            nm_data = F.relu(nm_data)
            nm_data = maxpool(nm_data, kernel_size=2, stride=2)
            w,b = vars[8], vars[9]
            nm_data = conv2d(nm_data, w, b, 1)
            w,b = vars[10], vars[11]
            running_mean, running_var = self.vars_bn[4], self.vars_bn[5]
            nm_data = F.batch_norm(nm_data, running_mean, running_var, weight=w, bias=b, training=bn_training)
            nm_data = F.relu(nm_data)
            if dataset == 'imagenet':
                nm_data = maxpool(nm_data, kernel_size=2, stride=2)
            nm_data = nm_data.view(nm_data.size(0), -1)
            w,b = vars[12], vars[13]
            fc_mask = F.sigmoid(F.linear(nm_data, w, b)).view(nm_data.size(0), -1)
            # =========== PREDICTION NETWORK ===========
            w,b = vars[14], vars[15]
            data = conv2d(data, w, b, 1)
            w,b = vars[16], vars[17]
            running_mean, running_var = self.vars_bn[6], self.vars_bn[7]
            data = F.batch_norm(data, running_mean, running_var, weight=w, bias=b, training=bn_training)
            data = F.relu(data)
            data = maxpool(data, kernel_size=2, stride=2)
            w,b = vars[18], vars[19]
            data = conv2d(data, w, b, 1)
            w,b = vars[20], vars[21]
            running_mean, running_var = self.vars_bn[8], self.vars_bn[9]
            data = F.batch_norm(data, running_mean, running_var, weight=w, bias=b, training=bn_training)
            data = F.relu(data)
            data = maxpool(data, kernel_size=2, stride=2)
            w,b = vars[22], vars[23]
            data = conv2d(data, w, b, 1)
            w,b, = vars[24], vars[25]
            running_mean, running_var = self.vars_bn[10], self.vars_bn[11]
            data = F.batch_norm(data, running_mean, running_var, weight=w, bias=b, training=bn_training)
            data = F.relu(data)
            if dataset == 'imagenet':
                data = maxpool(data, kernel_size=2, stride=2)
            data = data.view(data.size(0), -1)
            data = data*fc_mask
            w,b = vars[28], vars[29]
            data = F.linear(data, w, b)
            if self.aim is not None:
                rnd_idx, mech_choice = None, None
                if meta_train:
                    mech_choice = TOPK + 2
                    rng = default_rng()
                    rnd_idx = rng.choice(mech_choice, size=self.aim.topk, replace=False).tolist()
                data, logits, chosen = self.aim.aim_forward(data, vars[30], vars[31], vars[32], vars[33], vars[34],
                                                                stochastic=rnd_idx, mech_choice=mech_choice, threshold=True)
                data = F.normalize(data, p=2, dim=data.dim() - 1, eps=1e-12)
            w,b = vars[26], vars[27]
            data = F.linear(data, w, b)
        else:
            # =========== NEUROMODULATORY NETWORK ===========
            data = x

            w,b = vars[0], vars[1]
            data = conv2d(data, w, b, 1, 1)
            w,b = vars[2], vars[3]
            running_mean, running_var = self.vars_bn[0], self.vars_bn[1]
            data = F.batch_norm(data, running_mean, running_var, weight=w, bias=b, training=bn_training)
            data = F.relu(data)
            if dataset == 'imagenet':
                data = maxpool(data, kernel_size=2, stride=2)

            w,b = vars[4], vars[5]
            data = conv2d(data, w, b, 1, 1)
            w,b = vars[6], vars[7]
            running_mean, running_var = self.vars_bn[2], self.vars_bn[3]
            data = F.batch_norm(data, running_mean, running_var, weight=w, bias=b, training=bn_training)
            data = F.relu(data)
            if dataset != 'imagenet':
                data = maxpool(data, kernel_size=2, stride=2)

            w,b = vars[8], vars[9]
            data = conv2d(data, w, b, 1, 1)
            w,b = vars[10], vars[11]
            running_mean, running_var = self.vars_bn[4], self.vars_bn[5]
            data = F.batch_norm(data, running_mean, running_var, weight=w, bias=b, training=bn_training)
            data = F.relu(data)
            if dataset == 'imagenet':
                data = maxpool(data, kernel_size=2, stride=2)

            w,b = vars[12], vars[13]
            data = conv2d(data, w, b, 1, 1)
            w,b = vars[14], vars[15]
            running_mean, running_var = self.vars_bn[6], self.vars_bn[7]
            data = F.batch_norm(data, running_mean, running_var, weight=w, bias=b, training=bn_training)
            data = F.relu(data)
            if dataset != 'imagenet':
                data = maxpool(data, kernel_size=2, stride=2)

            w,b = vars[16], vars[17]
            data = conv2d(data, w, b, 1, 1)
            w,b = vars[18], vars[19]
            running_mean, running_var = self.vars_bn[8], self.vars_bn[9]
            data = F.batch_norm(data, running_mean, running_var, weight=w, bias=b, training=bn_training)
            data = F.relu(data)
            if dataset == 'imagenet':
                data = maxpool(data, kernel_size=2, stride=2)

            w,b = vars[20], vars[21]
            data = conv2d(data, w, b, 1, 1)
            w,b = vars[22], vars[23]
            running_mean, running_var = self.vars_bn[10], self.vars_bn[11]
            data = F.batch_norm(data, running_mean, running_var, weight=w, bias=b, training=bn_training)
            data = F.relu(data)
            data = avgpool(data, kernel_size=2, stride=2)
            data = data.view(data.size(0), -1)

            # =========== PREDICTION NETWORK ===========
            
            if self.aim is not None:
                data = data.view(data.size(0), -1)
                w,b = vars[24], vars[25]
                data = F.relu(F.linear(data, w, b))
                
                rnd_idx, mech_choice = None, None
                if meta_train:
                    mech_choice = TOPK + 2
                    rng = default_rng()
                    rnd_idx = rng.choice(mech_choice, size=self.aim.topk, replace=False).tolist()
                data, logits, chosen = self.aim.aim_forward(data, vars[28], vars[29], vars[30], vars[31], vars[32],
                                                              stochastic=rnd_idx, mech_choice=mech_choice, threshold=True)
                data = F.normalize(data, p=2, dim=data.dim() - 1, eps=1e-12)

            w, b = vars[26], vars[27]
            data = F.linear(data, w, b)
            
        if return_aim:
            return data, logits, chosen
        else:
            return data

    def zero_grad(self, vars=None):
        """
        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars
