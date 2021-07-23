import argparse
import logging
import random
import pickle
import math

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from scipy import stats

import datasets.datasetfactory as df
import datasets.task_sampler as ts
import model.learner as learner
import model.modelfactory as mf
from model.aim import blocked_grad
import utils
from experiment.experiment import experiment

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    total_clases = 10

    frozen_layers = []
    for temp in range(args.rln * 2):
        frozen_layers.append("vars." + str(temp))
    if args.treatment == 'ANML+AIM':
        frozen_layers.extend(["vars.28", "vars.29"])

    print("Frozen layers = %s", " ".join(frozen_layers))

    final_results_all = []
    temp_result = []
    total_clases = args.schedule

    if args.dataset == 'omniglot':
        classes = list(range(650))
    elif args.dataset == 'cifar100':
        classes = list(range(70, 100))
    elif args.dataset == 'imagenet':
        classes = list(range(64, 84))
    dataset = df.DatasetFactory.get_dataset(args.dataset, background=True, train=True, all=True)
    dataset_test = df.DatasetFactory.get_dataset(args.dataset, background=True, train=False, all=True)
    sampler = ts.SamplerFactory.get_sampler(args.dataset, classes, dataset, dataset_test)
    cudnn.benchmark = True
    
    for tot_class in total_clases:
        lr_list = [0.001, 0.0006, 0.0004, 0.00035, 0.0003, 0.00025, 0.0002, 0.00015, 0.0001, 0.00009, 0.00008, 0.00006, 0.00003, 0.00001]
        lr_all = []
        for lr_search in range(args.runs):

            keep = np.random.choice(classes, tot_class, replace=False).tolist()
            iterators_sorted = []
            for t in keep:
                iterators_sorted.append(sampler.sample_task([t]))
            iterator = sampler.sample_tasks(keep, not args.test)
            
            print(args)

            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

            results_mem_size = {}

            for mem_size in [args.memory]:
                max_acc = -10
                max_lr = -10
                for lr in lr_list:

                    print(lr)
                    maml = torch.load(args.model, map_location='cpu')
                    maml.treatment = args.treatment

                    if args.scratch:
                        config = mf.ModelFactory.get_model(args.treatment, args.dataset)
                        maml = learner.Learner(config)
                        # maml = MetaLearingClassification(args, config).to(device).net

                    maml = maml.to(device)

                    for name, param in maml.named_parameters():
                        param.learn = True

                    for name, param in maml.named_parameters():
                        if name in frozen_layers:
                            param.learn = False

                        else:
                            if args.reset:
                                w = nn.Parameter(torch.ones_like(param))
                                if len(w.shape) > 1:
                                    torch.nn.init.kaiming_normal_(w)
                                else:
                                    w = nn.Parameter(torch.zeros_like(param))
                                param.data = w
                                param.learn = True

                    if args.treatment == 'OML':
                        weights2reset = ["vars_14"]
                        #biases2reset = ["vars_15"]
                    else:
                        weight = maml.parameters()[26]
                        torch.nn.init.kaiming_normal_(weight)
                        weight = maml.parameters()[27]
                        torch.nn.init.zeros_(weight)
                       
                    filter_list = ["vars.{0}".format(v) for v in range(6)]

                    print("Filter list = %s" % ",".join(filter_list))

                    list_of_names = list(
                        map(lambda x: x[1], list(filter(lambda x: x[0] not in filter_list, maml.named_parameters()))))

                    list_of_params = list(filter(lambda x: x.learn, maml.parameters()))
                    list_of_names = list(filter(lambda x: x[1].learn, maml.named_parameters()))
                    
                    if args.scratch or args.no_freeze:
                        print("Empty filter list")
                        list_of_params = maml.parameters()
                    
                    for x in list_of_names:
                        print("Unfrozen layer = %s" % str(x[0]))
                    opt = torch.optim.Adam(list_of_params, lr=lr)

                    pbar = utils.ProgressBar()
                    seen_data = None
                    seen_targets = None
                    for j, it in enumerate(iterators_sorted):
                        for _ in range(args.epoch):
                            i = 0
                            for img, y in it:
                                img = img.to(device)
                                y = y.to(device)

                                logits = maml(img, meta_train=False, iterations=1, bn_training=False)
                                pred_q = (logits).argmax(dim=1)

                                opt.zero_grad()
                                loss = F.cross_entropy(logits, y)
                                loss.backward()
                                opt.step()
                                try: 
                                    seen_data = torch.cat([seen_data, img.cpu()], dim=0)
                                except:
                                    seen_data = img.cpu()
                                try: 
                                    seen_targets = torch.cat([seen_targets, y.cpu()], dim=0)
                                except:
                                    seen_targets = y.cpu()
                                i += 1
                                if i == 30:
                                    break
                        pbar.update(j, len(iterators_sorted))
                    batch_size = i

                    print("Result after one epoch for LR = %f" % lr)
                    correct = 0
                    total = 0
                    if args.test:
                        for img, target in iterator:
                            img = img.to(device)
                            target = target.to(device)
                            logits_q = maml(img, meta_train=False, iterations=1, bn_training=False)
                            logits_q = logits_q.squeeze(-1)

                            pred_q = (logits_q).argmax(dim=1)

                            correct += torch.eq(pred_q, target).sum().item()
                            total += img.size(0)
                    else:
                        for i in range(tot_class):
                            img = seen_data[i * batch_size:((i + 1) * batch_size)].to(device)
                            target = seen_targets[i * batch_size:((i + 1) * batch_size)].to(device)
                            logits_q = maml(img, meta_train=False, iterations=1, bn_training=False)
                            logits_q = logits_q.squeeze(-1)

                            pred_q = (logits_q).argmax(dim=1)

                            correct += torch.eq(pred_q, target).sum().item()
                            total += img.size(0)

                    print(str(correct / float(total)))
                    if (correct / float(total) > max_acc):
                        max_acc = correct / float(total)
                        max_lr = lr

                lr_all.append(max_lr)
                results_mem_size[mem_size] = (max_acc, max_lr)
                print("Final Max Result = %s" % str(max_acc))
            temp_result.append((tot_class, results_mem_size))
            print("A=  ", results_mem_size)
            print("Temp Results = %s" % str(results_mem_size))
            print("LR RESULTS = ", temp_result)

        best_lr = float(stats.mode(lr_all)[0][0])
        print("BEST LR %s= " % str(best_lr))

        for aoo in range(args.runs):

            keep = np.random.choice(classes, tot_class, replace=False).tolist()
            iterators_sorted = []
            for t in keep:
                iterators_sorted.append(sampler.sample_task([t]))
            iterator = sampler.sample_tasks(keep, not args.test)
            
            print(args)

            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

            results_mem_size = {}

            for mem_size in [args.memory]:
                max_acc = -10
                max_lr = -10

                lr = best_lr

                maml = torch.load(args.model, map_location='cpu')
                maml.treatment = args.treatment

                if args.scratch:
                    config = mf.ModelFactory.get_model("MRCL", args.dataset)
                    maml = learner.Learner(config)

                maml = maml.to(device)

                for name, param in maml.named_parameters():
                    param.learn = True

                for name, param in maml.named_parameters():
                    if name in frozen_layers:
                        param.learn = False
                    else:
                        if args.reset:
                            w = nn.Parameter(torch.ones_like(param))
                            if len(w.shape) > 1:
                                torch.nn.init.kaiming_normal_(w)
                            else:
                                w = nn.Parameter(torch.zeros_like(param))
                            param.data = w
                            param.learn = True

                if args.treatment == "OML":
                    weights2reset = ["vars_14"]
                    #biases2reset = ["vars_15"]
                else:
                    weight = maml.parameters()[26]
                    torch.nn.init.kaiming_normal_(weight)
                    weight = maml.parameters()[27]
                    torch.nn.init.zeros_(weight)
                
                correct = 0
                total = 0
                for img, target in iterator:
                    with torch.no_grad():

                        img = img.to(device)
                        target = target.to(device)
                        logits_q = maml(img, meta_train=False, iterations=1, bn_training=False)
                        pred_q = (logits_q).argmax(dim=1)
                        correct += torch.eq(pred_q, target).sum().item()
                        total += img.size(0)


                print("Pre-epoch accuracy %s" % str(correct / float(total)))

                filter_list = ["vars.{0}".format(v) for v in range(6)]

                print("Filter list = %s" % ",".join(filter_list))
               
                list_of_names = list(
                    map(lambda x: x[1], list(filter(lambda x: x[0] not in filter_list, maml.named_parameters()))))

                list_of_params = list(filter(lambda x: x.learn, maml.parameters()))
                list_of_names = list(filter(lambda x: x[1].learn, maml.named_parameters()))
                if args.scratch or args.no_freeze:
                    print("Empty filter list")
                    list_of_params = maml.parameters()
                
                for x in list_of_names:
                    print("Unfrozen layer = %s" % str(x[0]))
                opt = torch.optim.Adam(list_of_params, lr=lr)

                pbar = utils.ProgressBar()
                seen_data = None
                seen_targets = None
                for j, it in enumerate(iterators_sorted):
                    for _ in range(0, args.epoch):
                        i = 0
                        for img, y in it:
                            img = img.to(device)
                            y = y.to(device)
                            pred = maml(img, meta_train=False, iterations=1, bn_training=False)
                            opt.zero_grad()
                            loss = F.cross_entropy(pred, y)
                            loss.backward()
                            opt.step()
                            try: 
                                seen_data = torch.cat([seen_data, img.cpu()], dim=0)
                            except:
                                seen_data = img.cpu()
                            try: 
                                seen_targets = torch.cat([seen_targets, y.cpu()], dim=0)
                            except:
                                seen_targets = y.cpu()
                            i += 1
                            if i == 30:
                                break
                    pbar.update(j, len(iterators_sorted))
                batch_size = i

                print("Result after one epoch for LR = %f" % lr)
                
                correct = 0
                total = 0
                if args.test:
                    for img, target in iterator:
                        img = img.to(device)
                        target = target.to(device)
                        logits_q = maml(img, meta_train=False, iterations=1, bn_training=False)

                        pred_q = (logits_q).argmax(dim=1)

                        correct += torch.eq(pred_q, target).sum().item()
                        total += img.size(0)
                else:
                    for i in range(tot_class):
                        img = seen_data[i * batch_size:((i + 1) * batch_size)].to(device)
                        target = seen_targets[i * batch_size:((i + 1) * batch_size)].to(device)
                        logits_q = maml(img, meta_train=False, iterations=1, bn_training=False)
                        logits_q = logits_q.squeeze(-1)

                        pred_q = (logits_q).argmax(dim=1)

                        correct += torch.eq(pred_q, target).sum().item()
                        total += img.size(0)

                print(str(correct / float(total)))
                if (correct / float(total) > max_acc):
                    max_acc = correct / float(total)
                    max_lr = lr

                lr_list = [max_lr]
                results_mem_size[mem_size] = (max_acc, max_lr)
                print("Final Max Result = %s" % str(max_acc))
            final_results_all.append((tot_class, results_mem_size))
            print("A=  ", results_mem_size)
            print("Final results = %s", str(results_mem_size))
            print("FINAL RESULTS = ", final_results_all)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1)
    argparser.add_argument('--seed', type=int, help='epoch number', default=444)
    argparser.add_argument('--schedule', type=int, nargs='+', default=[15,30],
                        help='Decrease learning rate at these epochs.')
    argparser.add_argument('--memory', type=int, help='epoch number', default=0)
    argparser.add_argument('--model', type=str, help='epoch number', default="Neuromodulation_cifar100.net")
    argparser.add_argument('--scratch', action='store_true', default=False)
    argparser.add_argument('--dataset', help='Name of experiment', default="omniglot")
    argparser.add_argument('--dataset-path', help='Name of experiment', default=None)
    argparser.add_argument('--name', help='Name of experiment', default="evaluation")
    argparser.add_argument("--commit", action="store_true")
    argparser.add_argument("--no-freeze", action="store_true")
    argparser.add_argument('--reset', action="store_true")
    argparser.add_argument('--test', action="store_true")
    argparser.add_argument("--iid", action="store_true")
    argparser.add_argument("--rln", type=int, default=12)
    argparser.add_argument("--runs", type=int, default=10)
    argparser.add_argument('--treatment', help='OML+AIM or ANML+AIM', default='OML+AIM')

    args = argparser.parse_args()

    import os

    args.name = "/".join([args.dataset, "eval", str(args.epoch).replace(".", "_"), args.name])

    main(args)
