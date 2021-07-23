import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
import datasets.omniglot as om
import datasets.miniimagenet as mi


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name, train=True, path=None, background=True, all=False, element_per_class=600):

        if name == "omniglot":
            train_transform = transforms.Compose(
                [transforms.Resize((84, 84)),
                 transforms.ToTensor()])
            if path is None:
                return om.Omniglot("../data/omni", background=background, download=True, train=train,
                                   transform=train_transform, all=all)
            else:
                return om.Omniglot(path, download=True, background=train, transform=train_transform)

        elif name == "cifar100":
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            if train:
                train_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                train_transform = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])
            if path is None:
                return CIFAR100("/DATA/data_cifar100/", train=train, transform=train_transform)
            else:
                return CIFAR100(path, train=train, transform=train_transform)

        elif name == "imagenet":
            if path is None:
                return mi.MiniImagenet("/DATA/miniimagenet/", 'train' if train else 'test', elem_per_class=element_per_class, test=not train)
            else:
                return mi.MiniImagenet(path, 'train' if train else 'test', elem_per_class=element_per_class, test=not train)

        else:
            print("Unsupported Dataset")
            assert False
