import logging
import copy
import math
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import matplotlib.pyplot as plt

import model.learner as Learner
from model.aim import blocked_grad

class MetaLearingClassification(nn.Module):
    """
    MetaLearingClassification Learner
    """

    def __init__(self, args, config, treatment):

        super(MetaLearingClassification, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.update_step = args.update_step
        self.treatment = treatment

        self.net = Learner.Learner(config, treatment)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.meta_iteration = 0
        self.inputNM = True
        self.nodeNM = False
        self.layers_to_fix = []

    def reset_classifer(self, class_to_reset):
        if self.treatment == 'OML':
            weight = self.net.parameters()[14]
        else:
            weight = self.net.parameters()[26]
        #torch.nn.init.kaiming_normal_(weight[class_to_reset].unsqueeze(0))
        torch.nn.init.normal_(weight[class_to_reset].unsqueeze(0))

    def inner_update(self, x, fast_weights, y, bn_training):
        logits = self.net(x, fast_weights, bn_training=bn_training, meta_train=True, iterations=1)

        if fast_weights is None:
            fast_weights = self.net.parameters()

        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, fast_weights, allow_unused=True)

        if isinstance(self.update_lr, list):
            fast_weights = list(map(lambda p: p[1] - p[2] * p[0] if p[1].learn and p[0] is not None else p[1], zip(grad, fast_weights, self.update_lr)))
        else:
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0] if p[1].learn and p[0] is not None else p[1], zip(grad, fast_weights)))

        for params_old, params_new in zip(self.net.parameters(), fast_weights):
            params_new.learn = params_old.learn
        return fast_weights, logits

    def meta_loss(self, x, fast_weights, y, bn_training):
        logits = self.net(x, fast_weights, bn_training=bn_training, meta_train=True, iterations=1)
        loss_q = F.cross_entropy(logits, y)
        return loss_q, logits

    def eval_accuracy(self, logits, y):
        pred_q = F.softmax(logits, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y).sum().item()
        return correct

    def forward(self, d_traj_iterators, d_rand_iterator):
        """
        :param x_traj:   Input data of sampled trajectory
        :param y_traj:   Ground truth of the sampled trajectory
        :param x_rand:   Input data of the random batch of data
        :param y_rand:   Ground truth of the random batch of data
        :return:
        """
        for it in d_traj_iterators:
            i = 0
            for batch_idx, (data, targets) in enumerate(it):
                data, targets = data.cuda(), targets.cuda()
                fast_weights, logits = self.inner_update(data, None if i == 0 and batch_idx == 0 else fast_weights, targets, False)
                i += 1
                pred_q = F.softmax(logits, dim=1).argmax(dim=1)
                pred_q = logits.argmax(dim=1)
                if i == self.update_step:
                    break

        x_rand, y_rand = iter(d_rand_iterator).next()
        for it in d_traj_iterators:
            i = 0
            for batch_idx, (data, targets) in enumerate(it):
                try:
                    x_rand = torch.cat([x_rand, data], dim=0)
                    y_rand = torch.cat([y_rand, targets], dim=0)
                except:
                    x_rand = data
                    y_rand = targets
                i += 1
                if i == self.update_step:
                    break
        x_rand, y_rand = x_rand.cuda(), y_rand.cuda()
        meta_loss, logits = self.meta_loss(x_rand, fast_weights, y_rand, False)
        
        with torch.no_grad():
            pred_q = F.softmax(logits, dim=1).argmax(dim=1)
            pred_q = logits.argmax(dim=1)
            classification_accuracy = torch.eq(pred_q, y_rand).sum().item()
    
        self.net.zero_grad()

        NM_reset = False
    
        if NM_reset:

            layers_to_reset = list(range(14, 28))
            grads = torch.autograd.grad(meta_loss, self.net.parameters())
        
            for idx in range(len(self.net.parameters())):
                if idx in layers_to_reset:
                    self.net.parameters()[idx].grad = None
                else:
                    self.net.parameters()[idx].grad = grads[idx]
        else:
            meta_loss.backward()

        self.optimizer.step()
        
        classification_accuracy /= len(x_rand)
        
        self.meta_iteration += 1

        return classification_accuracy, meta_loss
