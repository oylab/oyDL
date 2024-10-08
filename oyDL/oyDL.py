from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import copy

cudnn.benchmark = True
plt.ion()  # interactive mode


def basetrans():
    from torchvision import transforms
    from torch import float

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(float),
        ]
    )


def trainset(data_dir, dataset=None):
    from oyDL import PathDataset, MyLazyDataset, pil_loader_grey
    from torch.utils.data import Dataset, Subset, WeightedRandomSampler, DataLoader
    from torch import float
    from torchvision import transforms
    from copy import deepcopy
    data_transforms = {
        "train": transforms.Compose(
            [
                # transforms.ToTensor(),
                # transforms.ConvertImageDtype(float),
                #transforms.CenterCrop(84),
                transforms.ColorJitter(brightness=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=(0, 180))
            ]
        ),
        "test": transforms.Compose(
            [
                # transforms.ToTensor(),
                # transforms.ConvertImageDtype(float),
                #transforms.CenterCrop(84),
            ]
        ),
        "val": transforms.Compose(
            [
                # transforms.ToTensor(),
                # transforms.ConvertImageDtype(float),
                #transforms.CenterCrop(84),
            ]
        ),
    }

    from torchvision import datasets
    if dataset==None:
        dataset = PathDataset(data_dir, loader=pil_loader_grey)

    _, counts = np.unique(dataset.targets, return_counts=True)
    class_weights = np.sum(counts) / counts


    # this is the full dataset! at this point we havent picked indices yet.
    dataloader_full = DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=8
    )

    # Create the index splits for training, validation and test
    train_size = 0.7
    num_train = len(dataset.indices)
    indices = list(range(num_train))
    split = int(np.floor(train_size * num_train))
    split2 = int(np.floor((train_size + (1 - train_size) / 2) * num_train))

    np.random.shuffle(indices)

    idx = {
        "train": indices[:split],
        "test": indices[split:split2],
        "val": indices[split2:],
    }

    image_datasets = {
        x: MyLazyDataset(dataset, indices=idx[x], transform = data_transforms[x]) for x in ["train", "test", "val"]
    }

    sample_weights = {
        x: [
            class_weights[t]
            for t in image_datasets[x].targets
        ]
        for x in ["train", "test", "val"]
    }

    # for class balancing
    sampler = {
        x: WeightedRandomSampler(sample_weights[x], len(sample_weights[x]))
        for x in ["train", "test", "val"]
    }

    batch_size = {"train": 32, "test": 400, "val": 400}

    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size[x],
            shuffle=False,
            num_workers=8,
            sampler=sampler[x],
        )
        for x in ["train", "test", "val"]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "test", "val"]}
    class_names = dataset.classes

    return dataloaders, dataloader_full, dataset_sizes, class_names


# define loader for grayscale tiff

from PIL import Image


def pil_loader_grey(path: str) -> Image.Image:
    #from cv2 import normalize

    img = Image.open(path)
    data = np.array(img)
    img = Image.fromarray(data - np.percentile(data, 0.01)) / (
        np.percentile(data, 99.5) - np.percentile(data, 0.01)
    )
    return img


from torchvision.datasets import ImageFolder




class PathDataset(ImageFolder):
    def __init__(self, paths, transform=basetrans(), loader=pil_loader_grey):
        from natsort import natsorted
        import itertools
        import glob

        if not (isinstance(paths, list) or isinstance(paths, np.ndarray)):
            paths = [paths]
        self.root = paths
        files = [glob.glob(path + "/**/*.tif", recursive=True) for path in paths]
        files = list(itertools.chain.from_iterable(files))
        self.files = files
        self.transform = transform
        self.labels = [filepath.split("/")[-2] for filepath in self.files]
        self.classes = natsorted(list(set(self.labels)))
        self.targets = [
            il for l in self.labels for il, L in enumerate(self.classes) if l == L
        ]
        self.indices = list(range(len(self.targets)))

        self.loader = loader

    def __getitem__(self, item):
        file = self.files[item]
        target = self.targets[item]
        label = self.labels[item]
        file = self.loader(file)
        if self.transform is not None:
            file = self.transform(file)
        return file, target, label

    def __len__(self):
        return len(self.files)


class MyLazyDataset(Dataset):
    def __init__(self, dataset, transform=None, indices = None):
        self.dataset = dataset
        self.root = dataset.root
        self.transform = transform
        if indices is None:
            indices = np.arange(len(dataset.targets))
        self.indices = indices

        self.targets = [self.dataset.targets[i] for i in indices]
        self.labels = [self.dataset.labels[i] for i in indices]
        self.classes = dataset.classes

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.dataset[self.indices[index]][0])
        else:
            x = self.dataset[self.indices[index]][0]

        y = self.dataset[self.indices[index]][1]
        z = self.dataset[self.indices[index]][2]
        return x, y, z

    def __len__(self):
        return len(self.indices)  


class TrackDataset(Dataset):
    '''
        Dataset from a oyLabImaging track object
    '''
    def __init__(self,track, transform=basetrans(),Channel = 'DIC N2',boxsize=64):
        self.transform = transform
        self.T = track.T
        self.loader = track.get_movie(Channel=Channel,boxsize=boxsize)
    def __getitem__(self, item):
        label = "?"
        file = self.loader[item]
        file = self.transform(file)
        return file, 0, '?'
    def __len__(self):
        return len(self.T)


class ResNet:
    def __init__(
        self, device, filename=None, network="resnet50", pretrained=True, inchans=1
    ):
        loaded = False

        if isinstance(filename, str):
            try:
                r = ResNet.load(filename)
                self.__dict__.update(r.__dict__)
                loaded = True
            except Exception as e:
                print(filename + " doesn't contain a model ",e)
        if not loaded:
            self.model_urls = {
                "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
                "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
                "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
                "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
                "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
                "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
                "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
                "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
                "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
            }
            # self.device = device
            self.train_data_root = []
            self.class_names = []
            self.filename = ""
            self._trained = False
            self.dataloaders = None
            self.model = self.resnet(
                network=network, pretrained=pretrained, inchans=inchans
            )
            loaded = True
        self.device = device

    # def __reduce__(self):
    #    return (self.__class__,(self.filename,))

    def save(self, filename=None):
        """
        save model
        """
        if filename == None:
            filename = self.filename
        assert is_path_exists_or_creatable(filename), filename + " is not a valit path"
        self.filename = filename
        import dill

        with open(filename, "wb") as dbfile:
            dill.dump(self, dbfile)
            print("saved model")

    @classmethod
    def load(cls, filename):
        """
        load model
        """
        assert is_path_exists_or_creatable(filename), filename + " is not a valit path"
        import dill

        with open(filename, "rb") as dbfile:
            r = dill.load(dbfile)
            print("loaded model")
        return r

    # function fo adjust model upload for grayscale
    def _load_pretrained(self, model, url, inchans=3):
        from torch.utils import model_zoo
        import torch.nn as nn

        state_dict = model_zoo.load_url(url)
        if inchans == 1:
            conv1_weight = state_dict["conv1.weight"]
            # state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
            state_dict["conv1.weight"] = conv1_weight[:, [0], :, :]
            model.conv1 = nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
        elif inchans != 3:
            assert False, "Invalid number of inchans for pretrained weights"
        # model.load_state_dict(state_dict,strict=False)
        try:
            model.load_state_dict(state_dict, strict=False)
        except RuntimeError as e:
            print('Ignoring "' + str(e) + '"')

    def resnet(self, network="resnet50", pretrained=False, inchans=3):
        """Constructs a ResNet-50 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        from torchvision.models import ResNet
        from torchvision.models.resnet import Bottleneck

        model = ResNet(Bottleneck, [3, 4, 6, 3])
        if pretrained:
            self._load_pretrained(model, self.model_urls[network], inchans=inchans)
        return model

    def train_model(
        self, dataloaders=None, criterion=None, optimizer=None, scheduler=None, num_epochs=25
    ):

        import time
        import copy
        from torch import set_grad_enabled, max, sum
        import torch.nn as nn
        import torch.optim as optim
        from torch.optim import lr_scheduler
        import warnings

        if self.dataloaders==None:
            if dataloaders!=None:
                self.dataloaders = dataloaders
            else:
                assert 'No dataloaders provided'
        elif self.dataloaders != dataloaders:
            if dataloaders!=None:
                warnings.warn('Model already trained with a different dataset! Proceed with caution')
                self.dataloaders = dataloaders
            else:
                dataloaders = self.dataloaders

            
        
        
        self.class_names = dataloaders["train"].dataset.classes
        [
            self.train_data_root.append(x)
            for x in dataloaders["train"].dataset.root
            if x not in self.train_data_root
        ]
        device = self.device
        model = self.model
        model = model.to(device)

        if criterion == None:
            criterion = nn.CrossEntropyLoss()
        if optimizer == None:
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        if scheduler == None:
            scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        else:
            scheduler = scheduler(optimizer)

        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_loss = 100
        train_stats={'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], }
    
        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs - 1))
            print("-" * 10)

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, targets, _ in dataloaders[phase]:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = max(outputs, 1)
                        loss = criterion(outputs, targets)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += sum(preds == targets.data)
                if phase == "train":
                    scheduler.step()

                epoch_loss = running_loss / dataloaders[phase].dataset.__len__()
                epoch_acc = (
                    running_corrects.double() / dataloaders[phase].dataset.__len__()
                )

                print(
                    "{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc)
                )
                if phase=='train':
                    train_stats['train_loss'].append(epoch_loss)
                    train_stats['train_acc'].append(epoch_acc.tolist())
                else:
                    train_stats['val_loss'].append(epoch_loss)
                    train_stats['val_acc'].append(epoch_acc.tolist())

                # deep copy the model
                if phase == "val" and epoch_loss < best_loss:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("Best val Acc: {:4f}".format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        self._trained = True
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        self.model = model
        return train_stats

    def visualize_model(self, loader, num_images=12, numcolumns=4):
        from torch import no_grad, max, sum

        device = self.device
        model = self.model
        model = model.to(device)
        class_names = self.class_names
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()

        with no_grad():
            for i, (inputs, targets, labels) in enumerate(loader):
                inputs = inputs.to(device)

                outputs = model(inputs)
                _, preds = max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(
                        int(np.ceil(num_images / numcolumns)), numcolumns, images_so_far
                    )
                    ax.axis("off")
                    try:
                        if labels[j] in class_names:
                            ax.set_title(
                                "predicted: {}".format(class_names[preds[j]])
                                + " ground truth: {}".format(labels[j])
                            )
                        else:
                            ax.set_title(
                                "predicted: {}".format(class_names[preds[j]])
                                + " ground truth: Unknown"
                            )
                    except:
                        ax.set_title(
                            "predicted: {}".format(class_names[preds[j]])
                            + " ground truth: Unknown"
                        )
                    imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)

    def getActivationsByType(self, dataloader_full, layertype=None):
        from functools import partial
        import collections
        import copy
        import torch.nn as nn
        from torch import no_grad, cat

        if layertype == None:
            layertype = nn.AdaptiveAvgPool2d

        device = self.device
        model_ft = copy.deepcopy(self.model)
        # a dictionary that keeps saving the activations as they come
        activations = collections.defaultdict(list)

        def save_activation(name, mod, inp, out):
            activations[name].append(out.cpu())

        # Registering hooks for all the Conv2d layers
        # Note: Hooks are called EVERY TIME the module performs a forward pass. For modules that are
        # called repeatedly at different stages of the forward pass (like RELUs), this will save different
        # activations. Editing the forward pass code to save activations is the way to go for these cases.
        handles = []
        for name, m in model_ft.named_modules():
            if type(m) == layertype:
                # partial to assign the layer name to each hook
                handles.append(m.register_forward_hook(partial(save_activation, name)))
        labelsssss = []
        # forward pass through the full dataset
        for (inputs, targets, labels) in dataloader_full:
            inputs = inputs.to(device)

            with no_grad():
                outputs = model_ft(inputs)
            labelsssss = labelsssss + ([int(p) for p in targets.cpu()])

        # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
        activations = {name: cat(outputs, 0) for name, outputs in activations.items()}
        labelsssss = np.array(labelsssss)

        # just print out the sizes of the saved activations as a sanity check
        for k, v in activations.items():
            print(k, v.size())

        # remove all hooks!
        [h.remove() for h in handles]
        return activations, labelsssss

    def get_logits_and_labels(self, dataloader_full):
        from torch import no_grad

        outcp = []
        labelsssss = []
        device = self.device
        model_ft = copy.deepcopy(self.model)
        model_ft.to(device)
        # m = nn.Softmax(dim=1)
        with no_grad():
            for i, (inputs, targets, _) in enumerate(dataloader_full):
                inputs = inputs.to(device)
                #targets = targets.to(device)
                outputs = model_ft(inputs)
                outcp = outcp + ([np.array(p) for p in outputs.cpu()])
                labelsssss = labelsssss + ([int(p) for p in targets])
        outcp = np.array(outcp)
        labelsssss = np.array(labelsssss)
        return (outcp, labelsssss)

    def get_softmax_and_labels(self, dataloader_full):
        import torch.nn as nn
        from torch import no_grad

        m = nn.Softmax(dim=1)
        outcp = []
        labelsssss = []
        device = self.device
        model_ft = copy.deepcopy(self.model)
        with no_grad():
            for i, (inputs, targets, _) in enumerate(dataloader_full):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = m(model_ft(inputs))
                outcp = outcp + ([np.array(p) for p in outputs.cpu()])
                labelsssss = labelsssss + ([int(p) for p in targets.cpu()])
        outcp = np.array(outcp)
        labelsssss = np.array(labelsssss)
        return (outcp, labelsssss)

    def get_predicts_and_labels(self, dataloader_full):
        import torch.nn as nn
        from torch import no_grad, max

        predicts = []
        labelsssss = []
        device = self.device
        model_ft = copy.deepcopy(self.model)
        with no_grad():
            for i, (inputs, targets, _) in enumerate(dataloader_full):
                inputs = inputs.to(device)
                targets = targets.to(device)
                _, preds = max(model_ft(inputs), 1)
                predicts = predicts + ([int(p) for p in preds.cpu()])
                labelsssss = labelsssss + ([int(p) for p in targets.cpu()])
        predicts = np.array(predicts)
        labelsssss = np.array(labelsssss)
        return (predicts, labelsssss)

    def ConfusionMatrix(self, dataloader):
        from torchmetrics import ConfusionMatrix
        from mlxtend.plotting import plot_confusion_matrix
        from torch import tensor
        import matplotlib

        class_names = self.class_names
        cmat = ConfusionMatrix(num_classes=len(class_names))
        preds, labels = self.get_predicts_and_labels(dataloader)
        cmat(tensor(preds), tensor(labels))

        cmat_tensor = cmat.compute()
        cmat = cmat_tensor.numpy()

        fig, ax = plot_confusion_matrix(
            conf_mat=cmat, class_names=class_names, show_normed=True
        )

    def classification_report(self, dataloader):
        #from torchmetrics import ConfusionMatrix
        from mlxtend.plotting import plot_confusion_matrix
        from torch import tensor
        import matplotlib

        class_names = self.class_names
        #cmat = ConfusionMatrix(num_classes=len(class_names))
        preds, labels = self.get_predicts_and_labels(dataloader)

        from sklearn import metrics
        report = metrics.classification_report(labels, preds, target_names=class_names, output_dict=True)
        print(metrics.classification_report(labels, preds, target_names=class_names))
        #print(report)
        return report


def is_pathname_valid(pathname: str) -> bool:
    """
    `True` if the passed pathname is a valid pathname for the current OS;
    `False` otherwise.
    """
    import errno, os, sys

    ERROR_INVALID_NAME = 123
    try:
        if not isinstance(pathname, str) or not pathname:
            return False
        _, pathname = os.path.splitdrive(pathname)
        root_dirname = (
            os.environ.get("HOMEDRIVE", "C:")
            if sys.platform == "win32"
            else os.path.sep
        )
        assert os.path.isdir(root_dirname)
        root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep
        for pathname_part in pathname.split(os.path.sep):
            try:
                os.lstat(root_dirname + pathname_part)
            except OSError as exc:
                if hasattr(exc, "winerror"):
                    if exc.winerror == ERROR_INVALID_NAME:
                        return False
                elif exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                    return False
    except TypeError as exc:
        return False
    else:
        return True


def is_path_creatable(pathname: str) -> bool:
    """
    `True` if the current user has sufficient permissions to create the passed
    pathname; `False` otherwise.
    """
    import os

    # Parent directory of the passed path. If empty, we substitute the current
    # working directory (CWD) instead.
    dirname = os.path.dirname(pathname) or os.getcwd()
    return os.access(dirname, os.W_OK)


def is_path_exists_or_creatable(pathname: str) -> bool:
    """
    `True` if the passed pathname is a valid pathname for the current OS _and_
    either currently exists or is hypothetically creatable; `False` otherwise.

    This function is guaranteed to _never_ raise exceptions.
    """
    import os

    try:
        # To prevent "os" module calls from raising undesirable exceptions on
        # invalid pathnames, is_pathname_valid() is explicitly called first.
        return is_pathname_valid(pathname) and (
            os.path.exists(pathname) or is_path_creatable(pathname)
        )
    # Report failure on non-fatal filesystem complaints (e.g., connection
    # timeouts, permissions issues) implying this path to be inaccessible. All
    # other exceptions are unrelated fatal issues and should not be caught here.
    except OSError:
        return False


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def InAxes(ax=None):
    # find indexes of plot points which are inside axes rectangle
    # by default works on the current axes, otherwise give an axes handle
    import matplotlib.pyplot as plt

    if ax == None:
        ax = plt.gca()

    h = ax.get_children()

    Xlim = ax.get_xlim()
    Ylim = ax.get_ylim()
    for hi in h:
        try:
            offs = hi.get_offsets().data
            J = np.where(
                (offs[:, 0] > Xlim[0])
                * (offs[:, 0] < Xlim[1])
                * (offs[:, 1] > Ylim[0])
                * (offs[:, 1] < Ylim[1])
            )
            J = J[0]
        except:
            continue
    return J


def manual_annotator(pth, class_names, dir_to_save=None, user="Teddy"):
    """ """
    from typing import List

    import numpy as np
    from magicgui import magicgui
    from magicgui.widgets import Container, PushButton
    from PIL import Image
    from datetime import datetime
    import textwrap

    if dir_to_save is None:
        dir_to_save = pth
    viewer = get_or_create_viewer()
    viewer.layers.clear()

    def fnamelistsFrompthcls(pth, class_names):
        import glob
        import os

        fnames = {
            cnm: glob.glob(
                os.path.join(pth, cnm) + "**" + os.path.sep + "*.[tT][iI][fF]",
                recursive=True,
            )
            for cnm in class_names
        }
        return fnames

    fnmlst = fnamelistsFrompthcls(pth, class_names)

    @magicgui(
        auto_call=True,
        pth={
            "widget_type": "Label",
            "value": "\n".join(textwrap.wrap(pth, 50, break_long_words=True)),
        },
    )
    def widget(pth: str):
        return ()

    def make_buttons(class_names: List[str]):
        buttons = []
        for class_name in class_names:
            btn = PushButton(text=class_name)

            @btn.clicked.connect
            def _(state, name=class_name):
                row_to_write = [
                    widget.fl,
                    widget.cls,
                    name,
                    datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                    user,
                ]
                _write_to_csv(
                    dir_to_save, row_to_write, csv_name="manual_classification.csv"
                )
                widget.cls, widget.fl = _load_next_image()

            buttons.append(btn)

        return buttons

    btn_group = Container()
    btn_group.extend(make_buttons(class_names))
    widget.insert(-1, btn_group)

    btn = PushButton(text="NEXT")
    widget.insert(-1, btn)

    @btn.clicked.connect
    def _on_next_clicked():
        widget.cls, widget.fl = _load_next_image()

    def _load_next_image():
        import random

        random_cls = random.choice(class_names)
        random_img = random.choice(fnmlst[random_cls])
        viewer.layers.clear()
        img = Image.open(random_img)
        viewer.add_image(np.array(img))
        return random_cls, random_img

    def _write_to_csv(pth, row, csv_name="manual_classification.csv"):
        import csv
        from os import path

        path_to_file = path.join(pth, csv_name)
        if not path.exists(path_to_file):
            header = ["Filename", "Ground Truth", "Manual Selection", "Time", "User"]
            f = open(path.join(pth, csv_name), "a")
            writer = csv.writer(f)
            writer.writerow(header)
        f = open(path.join(pth, csv_name), "a")
        writer = csv.writer(f)
        writer.writerow(row)
        f.close()

    widget.cls, widget.fl = _load_next_image()

    container = Container(layout="vertical")
    layout = container.native.layout()

    layout.addWidget(widget.native)  # adding native, because we're in Qt

    viewer.window.add_dock_widget(container)

    return layout


def get_or_create_viewer():
    import napari

    return napari.current_viewer() or napari.Viewer()
