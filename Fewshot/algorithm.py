# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
# ==============================================================================
# Source: https://github.com/hushell/sib_meta_learn


import os
import itertools
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from utils.outils import progress_bar, AverageMeter, accuracy, getCi
from utils.utils import to_device
import copy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.animation as animation
from collections import OrderedDict
from numpy.random import default_rng

nice_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "Times New Roman",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 11,
        "font.size": 11,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 4.5,
        # "legend.fontsize": 6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1,
        "lines.markersize": 2,
}

mpl.rcParams.update(nice_fonts)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
sns.set_context("paper", rc=nice_fonts)

width = 430.00462

def set_size(width, fraction=1):
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 0.5) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim




# sns.set(font_scale=.5)
rng = default_rng()

ims = []
masks = []
labels = []
att_scores = []
scores = []


class Algorithm:
    """
    Algorithm logic is implemented here with training and validation functions etc.

    :param args: experimental configurations
    :type args: EasyDict
    :param logger: logger
    :param netFeat: feature network
    :type netFeat: class `WideResNet` or `ConvNet_4_64`
    :param netSIB: Classifier/decoder
    :type netSIB: class `ClassifierSIB`
    :param netAIM: Attentive Independent Mechanism
    :type netAIM: class `AIM`
    :param optimizer: optimizer
    :type optimizer: torch.optim.SGD
    :param criterion: loss
    :type criterion: nn.CrossEntropyLoss
    """
    def __init__(self, args, logger, netFeat, netSIB, netAIM, optimizer, optimizer_aim, optimizer_hs, criterion):
        self.netFeat = netFeat
        self.netSIB = netSIB
        self.netAIM = netAIM
        self.optimizer = optimizer
        self.optimizer_aim = optimizer_aim
        self.optimizer_hs = optimizer_hs
        self.criterion = criterion
        self.useAIM = args.useAIM
        self.testAdd = args.testAdd

        self.nbIter = args.nbIter
        self.nStep = args.nStep
        self.aStep = args.aStep
        self.outDir = args.outDir
        self.nFeat = args.nFeat
        self.batchSize = args.batchSize
        self.nEpisode = args.nEpisode
        self.momentum = args.momentum
        self.weightDecay = args.weightDecay

        self.logger = logger
        self.device = torch.device('cuda' if args.cuda else 'cpu')

        # Load pretrained model
        if args.resumeFeatPth :
            if args.cuda:
                param = torch.load(args.resumeFeatPth)
            else:
                param = torch.load(args.resumeFeatPth, map_location='cpu')
            self.netFeat.load_state_dict(param)
            msg = '\nLoading netFeat from {}'.format(args.resumeFeatPth)
            self.logger.info(msg)

        if args.test:
            self.load_ckpt(args.ckptPth)


    def load_ckpt(self, ckptPth):
        """
        Load checkpoint from ckptPth.

        :param ckptPth: the path to the ckpt
        :type ckptPth: string
        """
        param = torch.load(ckptPth)
        self.netFeat.load_state_dict(param['netFeat'])
        self.netAIM.load_state_dict(param['netAIM'])
        self.netAIM.hs = param['netAIM_hs']
        self.netSIB.load_state_dict(param['SIB'])
        lr = param['lr']
        lr_hs = param['lr_hs']
        lr_aim = param['lr_aim']
        self.optimizer = torch.optim.SGD(itertools.chain(*[self.netSIB.parameters()]),
                                         lr,
                                         momentum=self.momentum,
                                         weight_decay=self.weightDecay,
                                         nesterov=True)
        self.optimizer_hs = torch.optim.SGD(self.netAIM.parameters(),
                                            lr_hs)
        self.optimizer_aim = torch.optim.SGD(self.netAIM.parameters(),
                                            lr_aim)
        msg = '\nLoading networks from {}'.format(ckptPth)
        self.logger.info(msg)


    def compute_grad_loss(self, clsScore, QueryLabel):
        """
        Compute the loss between true gradients and synthetic gradients.
        """
        # register hooks
        def require_nonleaf_grad(v):
            def hook(g):
                v.grad_nonleaf = g
            h = v.register_hook(hook)
            return h
        handle = require_nonleaf_grad(clsScore)

        loss = self.criterion(clsScore, QueryLabel)
        loss.backward(retain_graph=True) # need to backward again

        # remove hook
        handle.remove()

        gradLogit = self.netSIB.dni(clsScore) # B * n x nKnovel
        gradLoss = F.mse_loss(gradLogit, clsScore.grad_nonleaf.detach())

        return loss, gradLoss


    def validate(self, valLoader, lr=None, mode='val'):
        """
        Run one epoch on val-set.

        :param valLoader: the dataloader of val-set
        :type valLoader: class `ValLoader`
        :param float lr: learning rate for synthetic GD
        :param string mode: 'val' or 'train'
        """
        if mode == 'test':
            nEpisode = self.nEpisode
            self.logger.info('\n\nTest mode: randomly sample {:d} episodes...'.format(nEpisode))
        elif mode == 'val':
            nEpisode = len(valLoader)
            self.logger.info('\n\nValidation mode: pre-defined {:d} episodes...'.format(nEpisode))
            valLoader = iter(valLoader)
        else:
            raise ValueError('mode is wrong!')

        episodeAccLog = []
        top1 = AverageMeter()

        self.netFeat.eval()
        self.netAIM.train()
        #self.netSIB.eval() # set train mode, since updating bn helps to estimate better gradient

        if lr is None:
            lr = self.optimizer.param_groups[0]['lr']

        if mode == 'val':
            causal_plot_score = OrderedDict()
            causal_plot_mask = OrderedDict()

        for batchIdx in range(nEpisode):
            data = valLoader.getEpisode() if mode == 'test' else next(valLoader)
            data = to_device(data, self.device)

            SupportTensor, SupportLabel, QueryTensor, QueryLabel = \
                    data['SupportTensor'].squeeze(0), data['SupportLabel'].squeeze(0), \
                    data['QueryTensor'].squeeze(0), data['QueryLabel'].squeeze(0)
            if mode == 'val':
                LabelName = data['LabelName']


            with torch.no_grad():
                SupportFeat, QueryFeat = self.netFeat(SupportTensor), self.netFeat(QueryTensor)
            # Adapt AIMs to test task
            if self.useAIM and not self.testAdd:
                
                netAIM = copy.deepcopy(self.netAIM)
                netAIM.train()
                optimizer_hs = torch.optim.SGD(netAIM.parameters(),
                                                self.optimizer_hs.param_groups[0]['lr'])
                for i in range(self.aStep):
                    optimizer_hs.zero_grad()
                    SupportFeat_AIM = netAIM(SupportFeat.unsqueeze(1))
                    clsScore = self.netSIB(SupportFeat_AIM.unsqueeze(0), SupportLabel.unsqueeze(0), SupportFeat_AIM.unsqueeze(0), lr)
                    clsScore = clsScore.view(SupportFeat_AIM.shape[0], -1)
                    SupportLabel_hs = SupportLabel.view(-1)
                    loss = self.criterion(clsScore, SupportLabel_hs)
                    loss.backward()
                    optimizer_hs.step()
            else:
                netAIM = copy.deepcopy(self.netAIM)
                netAIM.train()
            # Infer using adapted AIMs
            if self.useAIM:
                SupportFeat, QueryFeat = netAIM(SupportFeat.unsqueeze(1)), netAIM(QueryFeat.unsqueeze(1), return_score=False)
            SupportFeat, QueryFeat, SupportLabel = \
                    SupportFeat.unsqueeze(0), QueryFeat.unsqueeze(0), SupportLabel.unsqueeze(0)
            clsScore = self.netSIB(SupportFeat, SupportLabel, QueryFeat, lr)
            clsScore = clsScore.view(QueryFeat.shape[0] * QueryFeat.shape[1], -1)
            QueryLabel = QueryLabel.view(-1)
            acc1 = accuracy(clsScore, QueryLabel, topk=(1,))
            top1.update(acc1[0].item(), clsScore.shape[0])

            msg = 'Top1: {:.3f}%'.format(top1.avg)
            progress_bar(batchIdx, nEpisode, msg)
            episodeAccLog.append(acc1[0].item())
            '''
            if mode == 'val' and self.useAIM:
                nQuery = int(QueryLabel.size(0) / len(LabelName))
                for i in range(5):
                    if LabelName[i][0] in causal_plot_score:
                        causal_plot_score[LabelName[i][0]] = (causal_plot_score[LabelName[i][0]] + score[i*nQuery:i*nQuery+nQuery].mean(dim=0).detach().cpu().numpy()) / 2.
                        causal_plot_mask[LabelName[i][0]] = (causal_plot_mask[LabelName[i][0]] + mask[i*nQuery:i*nQuery+nQuery].mean(dim=0).detach().cpu().numpy()) / 2.
                    else:
                        causal_plot_score[LabelName[i][0]] = score[i*nQuery:i*nQuery+nQuery].mean(dim=0).detach().cpu().numpy()
                        causal_plot_mask[LabelName[i][0]] = mask[i*nQuery:i*nQuery+nQuery].mean(dim=0).detach().cpu().numpy()
            '''
        '''
        if mode == 'val' and self.useAIM:
            ims.append(tuple(causal_plot_score.values()))
            masks.append(tuple(causal_plot_mask.values()))
            labels.append(tuple(causal_plot_score.keys()))
        '''
        mean, ci95 = getCi(episodeAccLog)
        self.logger.info('Final Perf with 95% confidence intervals: {:.3f}%, {:.3f}%'.format(mean, ci95))
        '''
        if self.useAIM:
            scores.append(score[0,:].cpu().data.numpy())
            att_scores.append(att_score[0,:,0].cpu().data.numpy())
        '''
        
        return mean, ci95


    def train(self, trainLoader, valLoader, lr=None, coeffGrad=0.0) :
        """
        Run one epoch on train-set.

        :param trainLoader: the dataloader of train-set
        :type trainLoader: class `TrainLoader`
        :param valLoader: the dataloader of val-set
        :type valLoader: class `ValLoader`
        :param float lr: learning rate for synthetic GD
        :param float coeffGrad: deprecated
        """
        bestAcc, ci = self.validate(valLoader, lr)
        self.logger.info('Acc improved over validation set from 0% ---> {:.3f} +- {:.3f}%'.format(bestAcc,ci))

        self.netSIB.train()
        self.netAIM.train()
        self.netFeat.eval()

        losses = AverageMeter()
        top1 = AverageMeter()
        history = {'trainLoss' : [], 'trainAcc' : [], 'valAcc' : []}

        stochastic = True
        # stochastic = False
        mech_choice = 10
        for episode in range(self.nbIter):
            data = trainLoader.getBatch()
            data = to_device(data, self.device)


            with torch.no_grad() :
                SupportTensor, SupportLabel, QueryTensor, QueryLabel = \
                        data['SupportTensor'], data['SupportLabel'], data['QueryTensor'], data['QueryLabel']
                nC, nH, nW = SupportTensor.shape[2:]

                SupportFeat = self.netFeat(SupportTensor.reshape(-1, nC, nH, nW))
                QueryFeat = self.netFeat(QueryTensor.reshape(-1, nC, nH, nW))

            if lr is None:
                lr = self.optimizer.param_groups[0]['lr']
            # Train hidden state of AIM
            self.optimizer_aim.zero_grad()
            self.optimizer.zero_grad()
            if self.useAIM and not self.testAdd:
                for i in range(self.aStep):
                    self.optimizer_hs.zero_grad()
                    if stochastic == True:
                        rnd_idx = rng.choice(mech_choice, size=self.netAIM.topk, replace=False).tolist()
                        SupportFeat_AIM = self.netAIM(SupportFeat.unsqueeze(1), stochastic=rnd_idx, mech_choice=mech_choice)
                    else:
                        SupportFeat_AIM = self.netAIM(SupportFeat.unsqueeze(1))
                    clsScore = self.netSIB(SupportFeat_AIM.unsqueeze(0), SupportLabel, SupportFeat_AIM.unsqueeze(0), lr)
                    clsScore = clsScore.view(SupportFeat_AIM.shape[0], -1)
                    
                    SupportLabel_hs = SupportLabel.view(-1)
                    loss = self.criterion(clsScore, SupportLabel_hs)
                    loss.backward()
                    self.optimizer_hs.step()
            # Train the rest of the network
            self.optimizer.zero_grad()
            self.optimizer_aim.zero_grad()
            if self.useAIM:
                if stochastic == True:
                    rnd_idx = rng.choice(mech_choice, size=self.netAIM.topk, replace=False).tolist()
                    SupportFeat, QueryFeat = self.netAIM(SupportFeat.unsqueeze(1), stochastic=rnd_idx, mech_choice=mech_choice), self.netAIM(QueryFeat.unsqueeze(1), stochastic=rnd_idx, mech_choice=mech_choice)
                else:
                    SupportFeat, QueryFeat = self.netAIM(SupportFeat.unsqueeze(1)), self.netAIM(QueryFeat.unsqueeze(1))
            clsScore = self.netSIB(SupportFeat.unsqueeze(0), SupportLabel, QueryFeat.unsqueeze(0), lr)
            clsScore = clsScore.view(QueryFeat.shape[0], -1)
            QueryLabel = QueryLabel.view(-1)
            loss = self.criterion(clsScore, QueryLabel)

            loss.backward()
            self.optimizer.step()
            self.optimizer_aim.step()

            acc1 = accuracy(clsScore, QueryLabel, topk=(1, ))
            top1.update(acc1[0].item(), clsScore.shape[0])
            losses.update(loss.item(), QueryFeat.shape[1])
            msg = 'Loss: {:.3f} | Top1: {:.3f}% '.format(losses.avg, top1.avg)
            progress_bar(episode, self.nbIter, msg)

            if episode % 1000 == 999 :
                acc, _ = self.validate(valLoader, lr)

                if acc > bestAcc :
                    msg = 'Acc improved over validation set from {:.3f}% ---> {:.3f}%'.format(bestAcc , acc)
                    self.logger.info(msg)

                    bestAcc = acc
                    self.logger.info('Saving Best')
                    torch.save({
                                'lr': lr,
                                'lr_hs': self.optimizer_hs.param_groups[0]['lr'],
                                'lr_aim': self.optimizer_aim.param_groups[0]['lr'],
                                'netFeat': self.netFeat.state_dict(),
                                'netAIM': self.netAIM.state_dict(),
                                'netAIM_hs': self.netAIM.hs,
                                'SIB': self.netSIB.state_dict(),
                                'nbStep': self.nStep,
                                }, os.path.join(self.outDir, 'netSIBBest.pth'))

                self.logger.info('Saving Last')
                torch.save({
                            'lr': lr,
                            'lr_hs': self.optimizer_hs.param_groups[0]['lr'],
                            'lr_aim': self.optimizer_aim.param_groups[0]['lr'],
                            'netFeat': self.netFeat.state_dict(),
                            'netAIM': self.netAIM.state_dict(),
                            'netAIM_hs': self.netAIM.hs,
                            'SIB': self.netSIB.state_dict(),
                            'nbStep': self.nStep,
                            }, os.path.join(self.outDir, 'netSIBLast.pth'))

                msg = 'Iter {:d}, Train Loss {:.3f}, Train Acc {:.3f}%, Val Acc {:.3f}%'.format(
                        episode, losses.avg, top1.avg, acc)
                self.logger.info(msg)
                history['trainLoss'].append(losses.avg)
                history['trainAcc'].append(top1.avg)
                history['valAcc'].append(acc)

                losses = AverageMeter()
                top1 = AverageMeter()
        def update_frame_ims(i):
            plt.clf()
            ax = sns.heatmap(ims[i], yticklabels=labels[i], square=True, cbar_kws={"shrink": 0.5})
            ax.set_title("Epoch: {}".format(i))
            plt.xlabel('Mechanisms')
            plt.ylabel('Class')
            plt.tight_layout()
        def update_frame_mask(i):
            plt.clf()
            ax = sns.heatmap(masks[i], yticklabels=labels[i], square=True, cbar_kws={"shrink": 0.5})
            ax.set_title("Epoch: {}".format(i))
            plt.xlabel('Mechanisms')
            plt.ylabel('Class')
            plt.tight_layout()
        '''
        if self.useAIM:
            global labels
            labels = [tuple(s if s != "maple_tree" else "maple tree" for s in tup) for tup in labels] # bug with latex renderer
            plt_name = "mini_wrn_5shot"

            ## Scores
            # write animation frames
            fig_scr = plt.figure()
            anim = animation.FuncAnimation(fig_scr, update_frame_ims, frames=len(ims)-1, interval=50)
            FFMpegWriter = animation.writers['ffmpeg']
            writer = FFMpegWriter(fps=15, bitrate=5000)
            anim.save("plots/score-{}.mp4".format(plt_name), writer=writer)
            
            # Attention scores
            # plt.figure()
            plt.figure(10, figsize=set_size(width * 0.5))
            plt.clf()
            plt.plot(att_scores)
            plt.title('Attention Scores')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.savefig('plots/att_score-epoch_{}.pdf'.format(plt_name), bbox_inches='tight')
            # plt.figure()
            plt.figure(11, figsize=set_size(width * 0.5))
            plt.clf()
            plt.plot(scores)
            plt.title('Softmax Scores')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.tight_layout()
            plt.savefig('plots/score-epoch_{}.pdf'.format(plt_name), bbox_inches='tight')
            
            # Difference in AIMs Activation
            plt.figure()
            plt.clf()
            ax = sns.heatmap(np.asarray(ims[-1]) - np.asarray(ims[0]), yticklabels=labels[0], square=True, cbar_kws={"shrink": 0.5})
            ax.set_title("Difference in AIMs Attention Weight")
            plt.xlabel('Mechanisms')
            plt.ylabel('Class')
            plt.tight_layout()
            plt.savefig('plots/score-diff_{}.pdf'.format(plt_name), bbox_inches='tight')
            # First
            plt.figure()
            plt.clf()
            ax = sns.heatmap(np.asarray(ims[0]), yticklabels=labels[0], square=True, cbar_kws={"shrink": 0.5})
            ax.set_title("AIMs Attention Weight (Epoch 0)")
            plt.xlabel('Mechanisms')
            plt.ylabel('Class')
            plt.tight_layout()
            plt.savefig('plots/score-first_{}.pdf'.format(plt_name), bbox_inches='tight')
            # Last
            plt.figure()
            plt.clf()
            ax = sns.heatmap(np.asarray(ims[-1]), yticklabels=labels[0], square=True, cbar_kws={"shrink": 0.5})
            ax.set_title("AIMs Attention Weight (Epoch 49)")
            plt.xlabel('Mechanisms')
            plt.ylabel('Class')
            plt.tight_layout()
            plt.savefig('plots/score-last_{}.pdf'.format(plt_name), bbox_inches='tight')

            ## Masks
            # write animation frames
            fig_msk = plt.figure()
            anim = animation.FuncAnimation(fig_msk, update_frame_mask, frames=len(masks)-1, interval=50)
            FFMpegWriter = animation.writers['ffmpeg']
            writer = FFMpegWriter(fps=15, bitrate=5000)
            anim.save("plots/mask-{}.mp4".format(plt_name), writer=writer)
            
            # Difference in AIMs Activation
            plt.figure()
            plt.clf()
            ax = sns.heatmap(np.asarray(masks[-1]) - np.asarray(masks[0]), yticklabels=labels[0], square=True, cbar_kws={"shrink": 0.5})
            ax.set_title("Difference in AIMs Attention Weight")
            plt.xlabel('Mechanisms')
            plt.ylabel('Class')
            plt.tight_layout()
            plt.savefig('plots/mask-diff_{}.pdf'.format(plt_name), bbox_inches='tight')
            # First
            plt.figure()
            plt.clf()
            ax = sns.heatmap(np.asarray(masks[0]), yticklabels=labels[0], square=True, cbar_kws={"shrink": 0.5})
            ax.set_title("AIMs Attention Weight (Epoch 0)")
            plt.xlabel('Mechanisms')
            plt.ylabel('Class')
            plt.tight_layout()
            plt.savefig('plots/mask-first_{}.pdf'.format(plt_name), bbox_inches='tight')
            # Last
            plt.figure()
            plt.clf()
            ax = sns.heatmap(np.asarray(masks[-1]), yticklabels=labels[0], square=True, cbar_kws={"shrink": 0.5})
            ax.set_title("AIMs Attention Weight (Epoch 49)")
            plt.xlabel('Mechanisms')
            plt.ylabel('Class')
            plt.tight_layout()
            plt.savefig('plots/mask-last_{}.pdf'.format(plt_name), bbox_inches='tight')
        '''

        return bestAcc, acc, history
