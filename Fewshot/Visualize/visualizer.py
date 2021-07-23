from __future__ import print_function
from visdom import Visdom
import numpy as np
import os
import sys
import ntpath
import time
from Visualize import util, html
import pdb
from subprocess import Popen, PIPE
# from scipy.misc import imresize
import matplotlib.pyplot as plt
import torch
import cv2
plt.switch_backend('agg')
if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError
from torchvision.utils import make_grid

class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        # self.use_html = isTrain and not opt.no_html
        # self.win_size = opt.display_winsize
        # self.name = opt.name
        self.port = opt.display_port
        # self.saved = False
        # self.vix = Visdom(port=8096)
        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(port=opt.display_port)
            # self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port)
            if not self.vis.check_connection():
                self.create_visdom_connections()
        
        # if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
        #     self.web_dir = os.path.join(opt.checkpoints_dir, 'web')
        #     self.img_dir = os.path.join(self.web_dir, 'images')
        #     print('create web directory %s...' % self.web_dir)
        #     util.mkdirs([self.web_dir, self.img_dir])

        # create a logging file to store training losses
        # self.log_name = os.path.join(opt.checkpoints_dir, 'loss_log.txt')
        # with open(self.log_name, "a") as log_file:
        #     now = time.strftime("%c")
        #     log_file.write('================ Training Loss (%s) ================\n' % now)

    # def reset(self):
    #     """Reset the self.saved status"""
    #     self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    # def plot_current_losses(self, epoch, counter_ratio, loss1, test_loss):
    #     """display the current losses on visdom display: dictionary of error labels and values

    #     Parameters:
    #         epoch (int)           -- current epoch
    #         counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
    #         losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
    #     """
    #     if not hasattr(self, 'plot_data'):
    #         self.plot_data = {'X': [], 'Y': [], 'GG': [],  'legend': ['Train Loss','Test Loss']}
    #         # self.plot_data = {'X': [], 'Y': [], 'legend': ['Train Loss']}
    #     # pdb.set_trace()
    #     self.plot_data['X'].append(epoch + counter_ratio)
    #     if self.opt.network =="MAML":
    #         self.plot_data['Y'].append(np.array([float(loss1[2]), float(test_loss)]))
    #     else:
    #         self.plot_data['Y'].append(np.array([float(loss1), float(test_loss)]))
    #     # pdb.set_trace()
    #     # self.plot_data['Y'].append(np.array([loss]))
        
    #     # pdb.set_trace()
    #     # try:
    #     if len(self.plot_data['Y']) == 1:
    #         self.vis.line(
    #             X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
    #             Y=np.array(self.plot_data['Y']),
    #             opts={
    #                 'title': self.opt.network + 'Net' + ' loss over time',
    #                 'legend': self.plot_data['legend'],
    #                 'xlabel': 'epoch',
    #                 'ylabel': 'loss'},
    #             win=self.display_id)
    #     else:
    #         self.vis.line(
    #             X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1).squeeze(),
    #             Y=np.array(self.plot_data['Y']).squeeze(),
    #             opts={
    #                 'title': self.name + ' loss over time',
    #                 'legend': self.plot_data['legend'],
    #                 'xlabel': 'epoch',
    #                 'ylabel': 'loss'},
    #             win=self.display_id)
    #     # pdb.set_trace()
    #     if float(loss1[1]) != 0:
    #         self.plot_data['GG'].append(np.array(float(loss1[1])))
    #         if self.opt.KL_loss:
    #             title = 'Gradient Gen Loss over time (KL)'
    #         else:
    #             title = 'Gradient Gen Loss over time (MSE)'
    #         self.vis.line(
    #             X=np.array(self.plot_data['X']), 
    #             Y=np.array(self.plot_data['GG']),
    #             opts={
    #             'title': title,
    #             'xlabel': 'epoch',
    #             'ylabel': 'loss'},
    #             win=self.display_id+50)

    #     # except VisdomExceptionBase:
    #     #     self.create_visdom_connections()

    # def plot_dataset_performance(self, epoch, counter_ratio, rmse, mae, me_rate, perf):
    #     if not 'Z' in self.plot_data:
    #         self.plot_data['Z'] = []
    #         self.plot_data['legend2'] = ['rmse', 'absolute me', 'me_rate']
    #     self.plot_data['Z'].append(np.array([float(rmse), float(mae), float(me_rate)]))

    #     if len(self.plot_data['Z']) == 1:
    #         name = self.opt.network + 'Net' + ' plubic set performance ('
    #         name += self.opt.public + ')  '
    #         name += '(%.2f/%.2f/%.2f)' % (perf[0][0], perf[1][0], perf[2][0])
    #         self.vis.line(
    #             X=np.stack([np.array(self.plot_data['X'])] *
    #                        len(self.plot_data['legend2']), 1),
    #             Y=np.array(self.plot_data['Z']),
    #             opts={
    #                 'title': name,
    #                 'legend': self.plot_data['legend2'],
    #                 'xlabel': 'epoch',
    #                 'ylabel': 'loss'},
    #             win=self.display_id+6)
    #     else:
    #         name = self.opt.network + 'Net' + ' plubic set performance ('
    #         name += self.opt.public + ')  '
    #         name += '(%.2f/%.2f/%.2f)' % (perf[0][0], perf[1][0], perf[2][0])
    #         self.vis.line(
    #             X=np.stack([np.array(self.plot_data['X'])] *
    #                        len(self.plot_data['legend2']), 1).squeeze(),
    #             Y=np.array(self.plot_data['Z']).squeeze(),
    #             opts={
    #                 'title': name,
    #                 'legend': self.plot_data['legend2'],
    #                 'xlabel': 'epoch',
    #                 'ylabel': 'loss'},
    #             win=self.display_id+6)
    #     # pdb.set_trace()

    # # losses: same format as |losses| of plot_current_losses
    # def print_current_losses(self, epoch, iters, loss, test_loss, dat):
    #     """print current losses on console; also save the losses to the disk

    #     Parameters:
    #         epoch (int) -- current epoch
    #         iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
    #         losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
    #         t_comp (float) -- computational time per data point (normalized by batch_size)
    #         t_data (float) -- data loading time per data point (normalized by batch_size)
    #     """
    #     message = '(epoch: %d, iters: %d) ' % (epoch, iters)
    #     if self.opt.network == "MAML":
    #         message += 'Loss: %.3e/%.3f/%.3f/%.3f  ' % (
    #             float(loss[0]), float(loss[1]), float(loss[2]), float(test_loss))
    #         # message += 'MSELoss: %.3e ' % float(loss[0])
    #         # message += 'OrdLoss: %.3f ' % float(loss[1])
    #     else:
    #         message += 'Loss: %.3e/%.3f  ' % (float(loss), float(test_loss))
    #         # message += 'Train Loss: %.3f ' % float(loss)
    #     message += 'Performance: %.3f/%.3f/%.3f                 ' % (float(dat[0]), float(dat[1]), float(dat[2]))

    #     print(message,'\t\t\r', end='')  # print the message
    #     with open(self.log_name, "a") as log_file:
    #         log_file.write('%s\n' % message)  # save the message

    # def plot_current_results(self, results, true_rPPG, epoch, use_debug_port, i, dataset_size):
    #     # pdb.set_trace()
    #     if(results.size(0) == 60):
    #         t = np.linspace(0, self.opt.win_size/30, self.opt.win_size, endpoint=False)
    #         t1 = t = t.reshape((1, self.opt.win_size))
    #     else:
    #         # pdb.set_trace()
    #         t = np.fft.rfftfreq(60, 1/float(30))*60
    #         # pdb.set_trace()
    #         t = t.reshape((1, t.size))
    #         t1 = np.linspace(0, self.opt.win_size/30, self.opt.win_size, endpoint=False)
    #         t1 = t1.reshape((1, self.opt.win_size))
    #     results = results.squeeze()
    #     true_rPPG = true_rPPG.squeeze()
    #     results = results.unsqueeze(0)
    #     true_rPPG = true_rPPG.unsqueeze(0)
    #     results = results.detach().numpy()
    #     true_rPPG = true_rPPG.detach().numpy()
    #     # pdb.set_trace()

    #     f1, axs1 = plt.subplots(2, 1, constrained_layout=True)
    #     f1.suptitle('rPPG results (epoch:%d)(iter: %d)' % (epoch, i))
    #     # print(real_B)
    #     axs1[0].plot(t, results, 'bo')
    #     axs1[0].set_title('prediction')
        
    #     axs1[1].plot(t, true_rPPG, 'ro')
    #     axs1[1].set_title('ground truth')
 
        # pdb.set_trace()
        

        # if not self.isTrain:
        #     self.vis.matplot(f1)
        # elif use_debug_port == 1:
        #     self.vis.matplot(f1, win=3)
        # elif use_debug_port == 0:
        #     self.vis.matplot(f1, win=2)
        # else:
        #     self.vis.matplot(f1, win=4)

        plt.close('all')
    

    def plot_ori_recon(self, original, recon, name, win, nrow=10):
        img = make_grid(original, normalize=True)
        recon = make_grid(recon, normalize=True)
        images = torch.stack([img, recon], dim=0).cpu()
        self.vis.images(images, opts=dict(title=name), nrow=nrow, win=self.opt.display_id+win)

    def plot_causal_feat(self, original, recon, cfeat, name, win, nrow=1):
        img = make_grid(original, normalize=True, nrow=1)
        recon = make_grid(recon, normalize=True, nrow=1)
        # cfeat = make_grid(cfeat.view(-1, *(cfeat.size()[2:])), normalize=True)
        cfeat = [make_grid(cfeat[:,i] , normalize=True, nrow=1) for i in range(cfeat.size(1))]
        images = torch.stack([img, recon, *cfeat], dim=0).cpu()
        self.vis.images(images, opts=dict(title=name), nrow=nrow, win=self.opt.display_id+win)

            
