import errno
import hashlib
import os
import os.path
import random
from collections import namedtuple
import logging
import sys, time
from torch.nn import functional as F
import numpy as np
import model.learner as Learner
transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')
import torch

class ProgressBar(object):
    def __init__(self):
        _, self.term_width = os.popen('stty size', 'r').read().split()
        self.term_width = int(self.term_width)

        self.last_time = time.time()
        self.begin_time = self.last_time

    def update(self, current, total, msg=None, TOTAL_BAR_LENGTH = 30.):
        if current == 0:
            self.begin_time = time.time()  # Reset for new bar.

        cur_len = int(TOTAL_BAR_LENGTH*current/total)
        rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

        sys.stdout.write(' [')
        for i in range(cur_len):
            sys.stdout.write('=')
        sys.stdout.write('>')
        for i in range(rest_len):
            sys.stdout.write('.')
        sys.stdout.write(']')

        cur_time = time.time()
        step_time = cur_time - self.last_time
        self.last_time = cur_time
        tot_time = cur_time - self.begin_time

        pred_time = tot_time / float(current + 1) * (float(total) - float(current + 1))

        L = []
        L.append('  %s' % self.format_time(tot_time))
        L.append(' | %s' % self.format_time(pred_time))
        if msg:
            L.append(' | ' + msg)

        msg = ''.join(L)
        sys.stdout.write(msg)
        for i in range(self.term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
            sys.stdout.write(' ')

        # Go back to the center of the bar.
        for i in range(self.term_width-int(TOTAL_BAR_LENGTH/2)+2):
            sys.stdout.write('\b')
        sys.stdout.write(' %d/%d ' % (current + 1, total))

        if current < total-1:
            sys.stdout.write('\r')
        else:
            sys.stdout.write('\n')
        sys.stdout.flush()

    def format_time(self, seconds):
        days = int(seconds / 3600/24)
        seconds = seconds - days*3600*24
        hours = int(seconds / 3600)
        seconds = seconds - hours*3600
        minutes = int(seconds / 60)
        seconds = seconds - minutes*60
        secondsf = int(seconds)
        seconds = seconds - secondsf
        millis = int(seconds*1000)

        f = ''
        i = 1
        if days > 0:
            f += str(days) + 'D'
            i += 1
        if hours > 0 and i <= 2:
            f += str(hours) + 'h'
            i += 1
        if minutes > 0 and i <= 2:
            f += str(minutes) + 'm'
            i += 1
        if secondsf > 0 and i <= 2:
            f += str(secondsf) + 's'
            i += 1
        if millis > 0 and i <= 2:
            f += str(millis) + 'ms'
            i += 1
        if f == '':
            f = '0ms'
        return f

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def freeze_layers(layers_to_freeze, maml, treatment):

    for name, param in maml.named_parameters():
        param.learn = True

    for name, param in maml.net.named_parameters():
        param.learn = True

    frozen_layers = []
    for temp in range(layers_to_freeze * 2):
        frozen_layers.append("net.vars." + str(temp))

    for name, param in maml.named_parameters():
        if name in frozen_layers:
            print("RLN layer %s" % str(name))
            param.learn = False

    list_of_names = list(filter(lambda x: x[1].learn, maml.named_parameters()))

    for a in list_of_names:
        print("TLN layer = %s" % a[0])

def log_accuracy(maml, iterator_test, device, step):
    correct = 0
    for img, target in iterator_test:
        with torch.no_grad():
            img = img.to(device)
            target = target.to(device)
            logits_q = maml.net(img, meta_train=False, iterations=1, bn_training=False)
            logits_q = logits_q.squeeze(-1)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct += torch.eq(pred_q, target).sum().item() / len(img)
    print("Test Accuracy = %s" % str(correct / len(iterator_test)))


def iterator_sorter(trainset, no_sort=True, random=True, pairs=False, classes=10):
    if no_sort:
        return trainset

    order = list(range(len(trainset.data)))
    np.random.shuffle(order)

    trainset.data = trainset.data[order]
    trainset.targets = np.array(trainset.targets)
    trainset.targets = trainset.targets[order]

    sorting_labels = np.copy(trainset.targets)
    sorting_keys = list(range(20, 20 + classes))
    if random:
        if not pairs:
            np.random.shuffle(sorting_keys)

    print("Order = ", [x - 20 for x in sorting_keys])
    for numb, key in enumerate(sorting_keys):
        if pairs:
            np.place(sorting_labels, sorting_labels == numb, key - (key % 2))
        else:
            np.place(sorting_labels, sorting_labels == numb, key)

    indices = np.argsort(sorting_labels)
    # print(indices)

    trainset.data = trainset.data[indices]
    trainset.targets = np.array(trainset.targets)
    trainset.targets = trainset.targets[indices]
    # print(trainset.targets)
    # print(trainset.targets )

    return trainset


def iterator_sorter_omni(trainset, no_sort=True, random=True, pairs=False, classes=10):
    return trainset


def remove_classes(trainset, to_keep):
    # trainset.data = trainset.data[order]
    trainset.targets = np.array(trainset.targets)
    # trainset.targets = trainset.targets[order]

    indices = np.zeros_like(trainset.targets)
    for a in to_keep:
        indices = indices + (trainset.targets == a).astype(int)
    indices = np.nonzero(indices)
    # logger.info(trainset.data[0])
    trainset.data = trainset.data[indices]
    trainset.targets = np.array(trainset.targets)
    trainset.targets = trainset.targets[indices]

    return trainset


def remove_classes_omni(trainset, to_keep):
    # trainset.data = trainset.data[order]
    trainset.targets = np.array(trainset.targets)
    # trainset.targets = trainset.targets[order]

    indices = np.zeros_like(trainset.targets)
    for a in to_keep:
        indices = indices + (trainset.targets == a).astype(int)
    indices = np.nonzero(indices)
    trainset.data = [trainset.data[i] for i in indices[0]]
    # trainset.data = trainset.data[indices]
    trainset.targets = np.array(trainset.targets)
    trainset.targets = trainset.targets[indices]

    return trainset


def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)


def list_dir(root, prefix=False):
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def resize_image(img, factor):
    '''
    :param img:
    :param factor:
    :return:
    '''
    img2 = np.zeros(np.array(img.shape) * factor)

    for a in range(0, img.shape[0]):
        for b in range(0, img.shape[1]):
            img2[a * factor:(a + 1) * factor, b * factor:(b + 1) * factor] = img[a, b]
    return img2
