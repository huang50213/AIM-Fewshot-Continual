from __future__ import print_function

import os
from os.path import join

import numpy as np
import torch.utils.data as data
from PIL import Image

import torchvision.transforms as transforms
from utils import download_url, check_integrity, list_dir, list_files


class Omniglot(data.Dataset):
    """`Omniglot <https://github.com/brendenlake/omniglot>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        background (bool, optional): If True, creates dataset from the "background" set, otherwise
            creates from the "evaluation" set. This terminology is defined by the authors.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset zip files from the internet and
            puts it in root directory. If the zip files are already downloaded, they are not
            downloaded again.
    """
    folder = 'omniglot-py'
    download_url_prefix = 'https://github.com/brendenlake/omniglot/raw/master/python'
    zips_md5 = {
        'images_background': '68d2efa1b9178cc56df9314c21c6e718',
        'images_evaluation': '6b91aef0f799c5bb55b94e3f2daec811'
    }

    def __init__(self, root, background=True,
                 transform=None, target_transform=None,
                 download=False, train=True, all=False):
        self.root = join(os.path.expanduser(root), self.folder)
        self.background = background
        self.transform = transform
        self.target_transform = target_transform
        self.images_cached = {}

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.target_folder = join(self.root, self._get_target_folder())
        self._alphabets = list_dir(self.target_folder)
        self._characters = sum([[join(a, c) for c in list_dir(join(self.target_folder, a))]
                                for a in self._alphabets], [])
        self._character_images = [[(image, idx) for image in list_files(join(self.target_folder, character), '.png')]
                                  for idx, character in enumerate(self._characters)]
        self._flat_character_images = sum(self._character_images, [])
        self.data = [x[0] for x in self._flat_character_images]
        self.targets = [x[1] for x in self._flat_character_images]
        self.data2 = []
        self.targets2 = []
        self.new_flat = []
        for a in range(int(len(self.targets) / 20)):
            start = a * 20
            if train:
                for b in range(start, start + 15):
                    self.data2.append(self.data[b])
                    self.targets2.append(self.targets[b])
                    self.new_flat.append(self._flat_character_images[b])
                    # print(self.targets[start+b])
            else:
                for b in range(start+15, start+20):#(start + 15, start + 20):
                    self.data2.append(self.data[b])
                    self.targets2.append(self.targets[b])
                    self.new_flat.append(self._flat_character_images[b])

        if all:
            pass
        else:
            self._flat_character_images = self.new_flat
            self.targets = self.targets2
            print(self.targets[0:30])
            self.data = self.data2

        print("Total classes = ", np.max(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name = self.data[index]
        character_class = self.targets[index]
        image_path = join(self.target_folder, self._characters[character_class], image_name)
        
        if image_path not in self.images_cached:

            image = Image.open(image_path, mode='r').convert('RGB')#L
            image = image.resize((28,28), resample=Image.LANCZOS)
            image = np.array(image, dtype=np.float32)
            normalize = transforms.Normalize(mean=[0.92206*256, 0.92206*256, 0.92206*256], std=[0.08426*256*256, 0.08426*256*256, 0.08426*256*256]) # adjust means and std of input data 
            self.transform = transforms.Compose([transforms.ToTensor(), normalize])
            if self.transform is not None:
                image = self.transform(image)

            self.images_cached[image_path] = image
        else:
            image = self.images_cached[image_path]

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, character_class

    def _cache_data(self):
        pass

    def _check_integrity(self):
        zip_filename = self._get_target_folder()
        if not check_integrity(join(self.root, zip_filename + '.zip'), self.zips_md5[zip_filename]):
            return False
        return True

    def download(self):
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        filename = self._get_target_folder()
        zip_filename = filename + '.zip'
        url = self.download_url_prefix + '/' + zip_filename
        download_url(url, self.root, zip_filename, self.zips_md5[filename])
        print('Extracting downloaded file: ' + join(self.root, zip_filename))
        with zipfile.ZipFile(join(self.root, zip_filename), 'r') as zip_file:
            zip_file.extractall(self.root)

    def _get_target_folder(self):
        return 'images_background' if self.background else 'images_evaluation'


if __name__ == '__main__':
    dset = Omniglot("../../data/omni")
    print(dset.targets[:16 * 15])
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    nice_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "Times New Roman",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 11,
        "font.size": 7,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 4.5,
        # "legend.fontsize": 6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8.5,
        "lines.linewidth": 1,
        "lines.markersize": 2,
    }

    mpl.rcParams.update(nice_fonts)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    width = 430.00462

    plt.figure(figsize=(6.0, 2.4))

    for c in range(16):
        plt.subplot(2, 8, c + 1)
        plt.title('Class %d' % c)
        image_path = join(dset.target_folder, dset._characters[dset.targets[c * 15]], dset.data[c * 15])
        img_to_show = np.array(Image.open(image_path, mode='r').convert('RGB'))
        image_path = join(dset.target_folder, dset._characters[dset.targets[c * 15 + 1]], dset.data[c * 15 + 1])
        img_to_show = np.concatenate((img_to_show, Image.open(image_path, mode='r').convert('RGB')), axis=0)
        plt.imshow(img_to_show)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('../plots/omniglot_example.pdf')