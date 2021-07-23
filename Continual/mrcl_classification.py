import argparse
import logging

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import datasets.datasetfactory as df
import datasets.task_sampler as ts
import model.modelfactory as mf
import utils.utils as utils
from experiment.experiment import experiment
from model.meta_learner import MetaLearingClassification

def main(args):
    utils.set_seed(args.seed)

    if args.dataset == 'omniglot':
        args.classes = list(range(963))
    elif args.dataset == 'cifar100':
        args.classes = list(range(70))
    elif args.dataset == 'imagenet':
        args.classes = list(range(64))

    dataset = df.DatasetFactory.get_dataset(args.dataset, background=True, train=True, all=True)
    dataset_test = df.DatasetFactory.get_dataset(args.dataset, background=True, train=False, all=True)

    iterator_train = torch.utils.data.DataLoader(dataset, batch_size=5,
                                                 shuffle=True, num_workers=1)
    iterator_test = torch.utils.data.DataLoader(dataset_test, batch_size=5,
                                                shuffle=True, num_workers=1)

    sampler = ts.SamplerFactory.get_sampler(args.dataset, args.classes, dataset, dataset_test)

    config = mf.ModelFactory.get_model(args.treatment, args.dataset)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    maml = MetaLearingClassification(args, config, args.treatment).to(device)
    
    if args.checkpoint:
        checkpoint = torch.load(args.saved_model, map_location='cpu')

        for idx in range(len(checkpoint)):
            maml.net.parameters()[idx].data = checkpoint.parameters()[idx].data

    maml = maml.to(device)

    utils.freeze_layers(args.rln, maml, args.treatment)
    cudnn.benchmark = True
    
    for step in range(args.steps):

        t1 = np.random.choice(args.classes, args.tasks, replace=False)

        d_traj_iterators = []
        for t in t1:
            d_traj_iterators.append(sampler.sample_task([t]))
            maml.reset_classifer(t)

        d_rand_iterator = sampler.get_complete_iterator()
        accs, loss = maml(d_traj_iterators, d_rand_iterator)

        if step % 40 == 0:
            print('step: %d / %d   training acc %s' % (step, args.steps, str(accs)))
        if step % 100 == 0 or step == args.steps - 1:
            torch.save(maml.net, '_'.join([args.dataset, args.treatment, str(step // 10000 * 10000) + '.net']))
        if step % 2000 == 0 and step != 0:
            utils.log_accuracy(maml, iterator_test, device, step)
            utils.log_accuracy(maml, iterator_train, device, step)

#
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--steps', type=int, help='epoch number', default=20000)
    argparser.add_argument('--treatment', help='Model type', default='OML+AIM')
    argparser.add_argument('--checkpoint', help='Use a checkpoint model', action='store_true')
    argparser.add_argument('--saved_model', help='Saved model to load', default='my_model.net')
    argparser.add_argument('--seed', type=int, help='Seed for random', default=9)
    argparser.add_argument('--tasks', type=int, help='meta batch size, namely task num', default=1)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-2)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=20)
    argparser.add_argument('--dataset', help='Name of experiment', default="omniglot")
    argparser.add_argument("--no-reset", action="store_true")
    argparser.add_argument("--rln", type=int, default=12)
    args = argparser.parse_args()

    print(args)
    main(args)
