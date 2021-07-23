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

import os
import itertools
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from utils.outils import progress_bar, AverageMeter, accuracy, getCi
from utils.utils import to_device

class CLUE:
    """
    CLUE logic is implemented here with training and validation functions etc.

    :param args: experimental configurations
    :type args: EasyDict
    :param logger: logger
    :param netFeat: feature network
    :type netFeat: class `WideResNet` or `ConvNet_4_64`
    :param netClass: Classifier/decoder
    :type netClass: class `Classifier`
    :param netRIM: Recurrent Independent Mechanisms (learns causality)
    :type netRIM: class `RIMCLUE`
    :param optimizer: optimizer
    :type optimizer: torch.optim.SGD
    :param criterion: loss
    :type criterion: nn.CrossEntropyLoss
    :param tstep: number of training steps for support and query set.
    :param astep: number of adaptation steps for causal-transductive learning
    """
    def __init__(self, args, logger, netFeat, netClass, netRIM, optimizer, optimizerRIM, criterion, criterionRIM):
        self.netFeat = netFeat
        self.netClass = netClass
        self.netRIM = netRIM
        self.optimizer = optimizer
        self.optimizerRIM = optimizerRIM
        self.criterion = criterion
        self.criterionRIM = criterionRIM

        self.nbIter = args.nbIter
        self.tStep = args.tStep
        self.aStep = args.aStep
        self.outDir = args.outDir
        self.nFeat = args.nFeat
        self.nUnit = args.rim_units
        self.nHidden = args.rim_hidden
        self.batchSize = args.batchSize
        self.nEpisode = args.nEpisode
        self.momentum = args.momentum
        self.weightDecay = args.weightDecay


        self.logger = logger
        self.device = torch.device('cuda' if args.cuda else 'cpu')

        # Load pretrained model
        if args.resumeFeatPth:
            if args.cuda:
                param = torch.load(args.resumeFeatPth)
            else:
                param = torch.load(args.resumeFeatPth, map_location='cpu')
            self.netFeat.load_state_dict(param)
            msg = '\nLoading netFeat from {}'.format(args.resumeFeatPth)
            self.logger.info(msg)
        if args.resumeRIMPth:
            if args.cuda:
                param = torch.load(args.resumeRIMPth)
            else:
                param = torch.load(args.resumeRIMPth, map_location='cpu')
            self.netRIM.load_state_dict(param)
            msg = '\nLoading netRIM from {}'.format(args.resumeRIMPth)
            self.logger.info(msg)
        # if args.resumeClassPth:
        #     if args.cuda:
        #         param = torch.load(args.resumeClassPth)
        #     else:
        #         param = torch.load(args.resumeClassPth, map_location='cpu')
        #     self.netClass.load_state_dict(param)
        #     msg = '\nLoading netClass from {}'.format(args.resumeClassPth)
        #     self.logger.info(msg)
        self.hs = torch.randn(self.nUnit, self.nHidden).cuda()
        self.cs = torch.randn(self.nUnit, self.nHidden).cuda()
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
        self.netClass.load_state_dict(param['netClass'])
        self.netRIM.load_state_dict(param['netRIM'])
        tStep = param['tStep']
        aStep = param['aStep']
        self.hs = param['hs']
        self.cs = param['cs']
        lr = param['lr']
        self.optimizer = torch.optim.SGD(
                                            [
                                                {"params": self.netFeat.parameters()},
                                                # {"params": self.netRIM.parameters()},
                                                {"params": self.netRIM.parameters(), "lr": 0.005},
                                                {"params": self.netClass.parameters()},
                                            ],
            # itertools.chain(*[self.netClass.parameters(),self.netFeat.parameters(),self.netRIM.parameters()]),
                                         lr,
                                         momentum=self.momentum,
                                         weight_decay=self.weightDecay,
                                         nesterov=True)
        msg = '\nLoading networks from {}'.format(ckptPth)
        self.logger.info(msg)



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
        # self.netFeat.train()
        self.netRIM.train()
        self.netClass.train()
        #self.netSIB.eval() # set train mode, since updating bn helps to estimate better gradient

        if lr is None:
            lr = self.optimizer.param_groups[0]['lr']

        #for batchIdx, data in enumerate(valLoader):
        for batchIdx in range(nEpisode):
            data = valLoader.getEpisode() if mode == 'test' else next(valLoader)
            data = to_device(data, self.device)

            SupportTensor, SupportLabel, QueryTensor, QueryLabel = \
                    data['SupportTensor'].squeeze(0), data['SupportLabel'].squeeze(0), \
                    data['QueryTensor'].squeeze(0), data['QueryLabel'].squeeze(0)

            with torch.no_grad():
                SupportLabel = SupportLabel.unsqueeze(0)

            # if batchIdx==0:
            #     self.hs, self.cs = self.hs.unsqueeze(0).repeat(QueryTensor.shape[0], 1, 1), \
            #         self.cs.unsqueeze(0).repeat(QueryTensor.shape[0], 1, 1)
            
            # QUERY
            # QueryFeat = self.netFeat(QueryTensor).view(self.batchSize, -1, self.nFeat)
            # self.hs, self.cs, gradFeat = self.netRIM(QueryFeat.permute(1,0,2).detach(), self.hs.detach(), self.cs.detach())
            # params = self.netFeat.transUpdate(QueryFeat.squeeze(0), gradFeat, lr=lr)

            # QueryFeat = self.netFeat(QueryTensor, params).view(self.batchSize, -1, self.nFeat)
            # SupportFeat = self.netFeat(SupportTensor, params).view(self.batchSize, -1, self.nFeat)
            QueryFeat = self.netFeat(QueryTensor).view(self.batchSize, -1, self.nFeat)
            SupportFeat = self.netFeat(SupportTensor).view(self.batchSize, -1, self.nFeat)
            clsScore = self.netClass(SupportFeat.detach(), QueryFeat, SupportLabel)
            clsScore = clsScore.view(QueryFeat.shape[0] * QueryFeat.shape[1], -1)
            QueryLabel = QueryLabel.view(-1)

            acc1 = accuracy(clsScore, QueryLabel, topk=(1,))
            top1.update(acc1[0].item(), clsScore.shape[0])

            msg = 'Top1: {:.3f}%'.format(top1.avg)
            progress_bar(batchIdx, nEpisode, msg)
            episodeAccLog.append(acc1[0].item())
            
            # Take average over batch across hidden and cell states
        # self.hs, self.cs = torch.mean(self.hs, dim=0), torch.mean(self.cs, dim=0)            

        mean, ci95 = getCi(episodeAccLog)
        self.logger.info('Final Perf with 95% confidence intervals: {:.3f}%, {:.3f}%'.format(mean, ci95))
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
        bestAcc, ci = self.validate(valLoader, lr=lr)
        self.logger.info('Acc improved over validation set from 0% ---> {:.3f} +- {:.3f}%'.format(bestAcc,ci))

        self.netFeat.eval()
        # self.netFeat.train()
        self.netClass.train()
        self.netRIM.train()

        losses = AverageMeter()
        top1 = AverageMeter()
        history = {'trainLoss' : [], 'trainAcc' : [], 'valAcc' : []}

        for episode in range(self.nbIter):
            data = trainLoader.getBatch()
            data = to_device(data, self.device)

            with torch.no_grad() :
                SupportTensor, SupportLabel, QueryTensor, QueryLabel = \
                        data['SupportTensor'], data['SupportLabel'], data['QueryTensor'], data['QueryLabel']
                nC, nH, nW = SupportTensor.shape[2:]
                SupportTensor, QueryTensor = SupportTensor.reshape(-1, nC, nH, nW), QueryTensor.reshape(-1, nC, nH, nW)
            if lr is None:
                lr = self.optimizer.param_groups[0]['lr']
            
            '''
            # Duplicate averaged hidden and cell states across batch dim of support
            # Query Set training
            # for _ in range(self.tStep):
            # for _ in range(self.aStep): # transductive learning using causal information
            # SupportFeat, QueryFeat = self.netFeat(SupportTensor, params=featParams), self.netFeat(QueryTensor, params=featParams)
            self.hs, self.cs = self.hs.unsqueeze(0).repeat(SupportTensor.shape[0], 1, 1), \
                 self.cs.unsqueeze(0).repeat(SupportTensor.shape[0], 1, 1)
            
            SupportFeat = self.netFeat(SupportTensor).view(self.batchSize, -1, self.nFeat)
            SupportFeat.retain_grad()
            clsScore = self.netClass(SupportFeat.detach(), SupportFeat, SupportLabel).view(SupportFeat.shape[0] * SupportFeat.shape[1], -1)
            loss = self.criterion(clsScore, SupportLabel.view(-1))
            # print('l296:', 'loss', loss.item(), 'clsScore', clsScore.sum().item(), 'SupportLabel', SupportLabel.sum().item())
            self.optimizer.zero_grad()
            loss.backward()

            self.hs, self.cs, gradFeat = self.netRIM(SupportFeat.permute(1,0,2).detach(), self.hs.detach(), self.cs.detach())
            lossRIM = self.criterionRIM(gradFeat, SupportFeat.grad.squeeze(0))
            # print('l302:', 'lossRIM', lossRIM.item(), 'gradFeat', gradFeat.sum().item(), 'supportFeat.grad', SupportFeat.grad.squeeze(0).sum().item())
            lossRIM.backward()            
            # print('rimlinear', torch.sum(self.netRIM.gradFeatLinear.weight).cpu().item(), 'rimlinear_grad', torch.sum(self.netRIM.gradFeatLinear.weight.grad).cpu().item())
            self.optimizerRIM.step()
            # Take average over batch across hidden and cell states
            self.hs, self.cs = torch.mean(self.hs, dim=0), torch.mean(self.cs, dim=0)   
            # Duplicate averaged hidden and cell states across batch dim of query
            self.hs, self.cs = self.hs.unsqueeze(0).repeat(QueryTensor.shape[0], 1, 1), \
                 self.cs.unsqueeze(0).repeat(QueryTensor.shape[0], 1, 1)
            
            # QUERY
            QueryFeat = self.netFeat(QueryTensor).view(self.batchSize, -1, self.nFeat)
            # if (QueryFeat!=QueryFeat).any():
            #     import pdb; pdb.set_trace()
            self.hs, self.cs, gradFeat = self.netRIM(QueryFeat.permute(1,0,2).detach(), self.hs.detach(), self.cs.detach())
            params = self.netFeat.transUpdate(QueryFeat.squeeze(0), gradFeat.detach(), lr=lr)
            # self.netFeat.set_params(params)
            '''

            # self.netFeat.transUpdate(SupportFeat.squeeze(0), gradFeat, lr=lr)
            SupportFeat = self.netFeat(SupportTensor).view(self.batchSize, -1, self.nFeat)
            QueryFeat = self.netFeat(QueryTensor).view(self.batchSize, -1, self.nFeat)
            # SupportFeat = self.netFeat(SupportTensor, params).view(self.batchSize, -1, self.nFeat)
            # QueryFeat = self.netFeat(QueryTensor, params).view(self.batchSize, -1, self.nFeat)
            # QueryFeat.retain_grad()
            clsScore = self.netClass(SupportFeat.detach(), QueryFeat.detach(), SupportLabel)
            clsScore = clsScore.view(QueryFeat.shape[0] * QueryFeat.shape[1], -1)
            QueryLabel = QueryLabel.view(-1)
            loss = self.criterion(clsScore, QueryLabel)
            # print('l340:', 'loss', loss.item(), 'clsScore', clsScore.sum().item(), 'QueryLbel', QueryLabel.sum().item())
            self.optimizer.zero_grad()
            loss.backward()
            # lossRIM = self.criterionRIM(gradFeat, QueryFeat.grad.squeeze(0).detach())
            # lossRIM.backward()
            # print('l345', 'lossRIM', lossRIM.item(), 'gradFeat', gradFeat.sum().item(), 'QueryFeat.grad', QueryFeat.grad.squeeze(0).sum().item())
            self.optimizer.step()
            # self.optimizerRIM.step()
            # SupportFeat = self.netFeat(SupportTensor).view(self.batchSize, -1, self.nFeat)
            # clsScore = self.netClass.trans_inference(SupportFeat.detach(), QueryFeat, SupportLabel, gradClass)
            # self.netFeat.set_params(featParams)
            # print('rimlinear', self.netRIM.gradFeatLinear.weight, 'rimlinear_grad', self.netRIM.gradFeatLinear.weight.grad)
            # print('key', self.netRIM.key.weight, 'key_grad', self.netRIM.key.weight.grad)
            # print('rimlinear', torch.sum(self.netRIM.gradFeatLinear.weight).cpu().item(), 'rimlinear_grad', torch.sum(self.netRIM.gradFeatLinear.weight.grad).cpu().item())
            
            # Take average over batch across hidden and cell states
            # self.hs, self.cs = torch.mean(self.hs, dim=0), torch.mean(self.cs, dim=0)   

            acc1 = accuracy(clsScore, QueryLabel, topk=(1, ))
            top1.update(acc1[0].item(), clsScore.shape[0])
            losses.update(loss.item(), QueryFeat.shape[1])
            msg = 'Loss: {:.3f} | Top1: {:.3f}% '.format(losses.avg, top1.avg)
            progress_bar(episode, self.nbIter, msg)

            if episode % 1000 == 999 :
                acc, _ = self.validate(valLoader, lr=lr)

                if acc > bestAcc :
                    msg = 'Acc improved over validation set from {:.3f}% ---> {:.3f}%'.format(bestAcc , acc)
                    self.logger.info(msg)
                    hs, cs = torch.mean(self.hs, dim=0), torch.mean(self.cs, dim=0)
                    bestAcc = acc
                    self.logger.info('Saving Best')
                    torch.save({
                                'lr': lr,
                                'netFeat': self.netFeat.state_dict(),
                                'netClass': self.netClass.state_dict(),
                                'netRIM': self.netRIM.state_dict(),
                                'tStep': self.tStep,
                                'aStep': self.aStep,
                                'hs': self.hs,
                                'cs': self.cs,
                                }, os.path.join(self.outDir, 'netCLUEBest.pth'))

                self.logger.info('Saving Last')
                torch.save({
                            'lr': lr,
                            'netFeat': self.netFeat.state_dict(),
                            'netClass': self.netClass.state_dict(),
                            'netRIM': self.netRIM.state_dict(),
                            'tStep': self.tStep,
                            'aStep': self.aStep,
                            'hs': self.hs,
                            'cs': self.cs,
                            }, os.path.join(self.outDir, 'netCLUELast.pth'))

                msg = 'Iter {:d}, Train Loss {:.3f}, Train Acc {:.3f}%, Val Acc {:.3f}%'.format(
                        episode, losses.avg, top1.avg, acc)
                self.logger.info(msg)
                history['trainLoss'].append(losses.avg)
                history['trainAcc'].append(top1.avg)
                history['valAcc'].append(acc)

                losses = AverageMeter()
                top1 = AverageMeter()

        return bestAcc, acc, history




'''
# Duplicate averaged hidden and cell states across batch dim of support
self.hs, self.cs = self.hs.unsqueeze(0).repeat(SupportTensor.shape[0], 1, 1), \
        self.cs.unsqueeze(0).repeat(SupportTensor.shape[0], 1, 1)

SupportFeat = self.netFeat(SupportTensor).view(self.batchSize, -1, self.nFeat)
self.hs, self.cs, gradFeat = self.netRIM(SupportFeat.permute(1,0,2).detach(), self.hs.detach(), self.cs.detach())
params = self.netFeat.transUpdate(SupportFeat.squeeze(0), gradFeat.detach(), lr=lr)
# self.netFeat.set_params(params)
SupportFeat = self.netFeat(SupportTensor, params).view(self.batchSize, -1, self.nFeat)
SupportFeat.retain_grad()
clsScore = self.netClass(SupportFeat.detach(), SupportFeat, SupportLabel).view(SupportFeat.shape[0] * SupportFeat.shape[1], -1)
loss = self.criterion(clsScore, SupportLabel.view(-1))
# print('l185:', 'loss', loss.item(), 'clsScore', clsScore.sum().item(), 'SupportLabel', SupportLabel.sum().item())
self.optimizer.zero_grad()
loss.backward()
lossRIM = self.criterionRIM(gradFeat, SupportFeat.grad.squeeze(0).detach())
# print('l189', 'lossRIM', lossRIM.item(), 'gradFeat', gradFeat.sum().item(), 'supportFeat.grad', SupportFeat.grad.squeeze(0).sum().item())
lossRIM.backward()
# print('rimlinear', self.netRIM.gradFeatLinear.weight, 'rimlinear_grad', self.netRIM.gradFeatLinear.weight.grad)
self.optimizerRIM.step()
# if (SupportFeat!=SupportFeat).any():
#     import pdb; pdb.set_trace()


# Take average over batch across hidden and cell states
# Duplicate averaged hidden and cell states across batch dim of query
self.hs, self.cs = torch.mean(self.hs, dim=0), torch.mean(self.cs, dim=0)   
'''


'''
for _ in range(self.tStep):
    SupportFeat = self.netFeat(SupportTensor).view(self.batchSize, -1, self.nFeat)
    self.hs, self.cs, gradFeat = self.netRIM(SupportFeat.permute(1,0,2).detach(), self.hs.detach(), self.cs.detach())
    # self.hs, self.cs, gradFeat, gradClass = self.netRIM(SupportFeat.permute(1,0,2).detach(), self.hs.detach(), self.cs.detach())
    featParams = self.netFeat.transUpdate(SupportFeat.squeeze(0), gradFeat)
    # classParams = self.netClass.transUpdate(SupportFeat.detach(), SupportFeat.detach(), SupportLabel, gradClass)
    SupportFeat = self.netFeat(SupportTensor, featParams).view(self.batchSize, -1, self.nFeat)
    # clsScore = self.netClass.apply_classification_weights(SupportFeat, classParams)
    clsScore = self.netClass(SupportFeat, SupportFeat, SupportLabel)
    clsScore = clsScore.view(SupportFeat.shape[1], -1) # nT x nK
    loss = self.criterion(clsScore, SupportLabel.view(-1))
    self.optimizer.zero_grad()
    loss.backward()
    self.netFeat.set_params(featParams)
    self.optimizer.step()
    # transductive update of feature extractor and classifier using causal information form RIMs
    # self.optimizer.zero_grad()
    # SupportFeat = self.netFeat(SupportTensor)
    # print("Support set", torch.cuda.memory_allocated(0))
'''


            # if (SupportFeat!=SupportFeat).any():
            #     import pdb; pdb.set_trace()
            # self.hs, self.cs, gradFeat, gradClass = self.netRIM(QueryFeat.permute(1,0,2).detach(), self.hs.detach(), self.cs.detach())
            # transductive update of feature extractor and classifier using causal information form RIMs
            
            # SupportLabel = SupportLabel.view(-1)
            # print('rimlinear', self.netRIM.gradFeatLinear.weight, 'rimlinear_grad', self.netRIM.gradFeatLinear.weight.grad)
            # print('rimcombine', self.netRIM.gradFeatCombineRIMs.weight, 'rimcombine_grad', self.netRIM.gradFeatCombineRIMs.weight.grad)

            # print('query feat_params', featParams)
            # print("Query set adaptation", torch.cuda.memory_allocated(0))
