from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import copy

cudnn.benchmark = True
plt.ion()   # interactive mode


def basetrans():
    from torchvision import transforms
    from torch import float
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(float),
    ])

def trainset(data_dir):
    from oyDL import PathDataset, MyLazyDataset, pil_loader_grey
    from torch.utils.data import Dataset, Subset, WeightedRandomSampler, DataLoader
    from torch import float
    from torchvision import transforms
    data_transforms = {
        'train': transforms.Compose([
            #transforms.ToTensor(),
            #transforms.ConvertImageDtype(float),
            transforms.ColorJitter(brightness=.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]),
        'test': transforms.Compose([
            #transforms.ToTensor(),
            #transforms.ConvertImageDtype(float),
        ]),
        'val': transforms.Compose([
            #transforms.ToTensor(),
            #transforms.ConvertImageDtype(float),
        ]),
    }

    from torchvision import datasets
    dataset = PathDataset(data_dir, loader=pil_loader_grey)
    _,counts = np.unique(dataset.targets, return_counts=True)
    class_weights = np.sum(counts)/counts

    trans_dataset = {x: MyLazyDataset(dataset,data_transforms[x])
                      for x in ['train', 'test', 'val']}

    # this is the full dataset! at this point we havent picked indices yet.
    dataloader_full = DataLoader(trans_dataset['test'], batch_size=64,
                                                  shuffle=False, num_workers=8)

    # Create the index splits for training, validation and test
    train_size = 0.7
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(train_size * num_train))
    split2 = int(np.floor((train_size+(1-train_size)/2) * num_train))
    np.random.shuffle(indices)
    idx = {'train': indices[:split],'test': indices[split:split2],'val': indices[split2:]}

    image_datasets = {x: Subset(trans_dataset[x],indices=idx[x])
                      for x in ['train', 'test', 'val']}

    sample_weights = {x: [class_weights[image_datasets[x].dataset.dataset.targets[i]] for i in image_datasets[x].indices]
    for x in ['train', 'test', 'val']}

    # for class balancing
    sampler = {x: WeightedRandomSampler(sample_weights[x], len(sample_weights[x]))
    for x in ['train', 'test', 'val']}

    batch_size = {'train': 32, 'test': 200, 'val': 200}


    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size[x],
                                                 shuffle=False, num_workers=8, sampler=sampler[x])
                      for x in ['train', 'test', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test','val']}
    class_names = dataset.classes

    return dataloaders,dataloader_full, dataset_sizes, class_names

# define loader for grayscale tiff

from PIL import Image
def pil_loader_grey(path: str) -> Image.Image:
    from cv2 import normalize
    img = Image.open(path)
    data = np.array(img)
    img = Image.fromarray(data-np.percentile(data, 0.01))/(np.percentile(data, 99.5)-np.percentile(data, 0.01))
    return img


from torchvision.datasets import ImageFolder
class PathDataset(ImageFolder):
    def __init__(self, paths, transform=basetrans(),loader=pil_loader_grey):
        from natsort import natsorted
        import itertools
        import glob


        if not (isinstance(paths, list) or isinstance(paths, np.ndarray)):
                paths = [paths]
        self.root=paths
        files = [glob.glob(path+'*.tif') for path in paths]
        files = list(itertools.chain.from_iterable(files))
        self.files = files
        self.transform = transform
        self.labels = [filepath.split('/')[-2] for filepath in self.files]
        self.classes = natsorted(list(set(self.labels)))
        self.targets = [il for l in self.labels for il, L in enumerate(self.classes) if l==L]

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
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        y = self.dataset[index][1]
        z = self.dataset[index][2]
        return x, y ,z
    def __len__(self):
        return len(self.dataset)


class ResNet:
    def __init__(self, device,network='resnet50', pretrained=True,inchans=1):
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
        self.device = device
        self.data_root=[]
        self.class_names = []
        self._trained = False
        self.model = self.resnet(network=network, pretrained=pretrained,inchans=inchans)

    #function fo adjust model upload for grayscale
    def _load_pretrained(self, model, url, inchans=3):
        from torch.utils import model_zoo
        import torch.nn as nn
        state_dict = model_zoo.load_url(url)
        if inchans == 1:
            conv1_weight = state_dict['conv1.weight']
            #state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
            state_dict['conv1.weight'] = conv1_weight[:,[0],:,:]
            model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        elif inchans != 3:
            assert False, "Invalid number of inchans for pretrained weights"
        #model.load_state_dict(state_dict,strict=False)
        try:
            model.load_state_dict(state_dict,strict=False)
        except RuntimeError as e:
            print('Ignoring "' + str(e) + '"')


    def resnet(self,network='resnet50',pretrained=False, inchans=3):
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

    def train_model(self, dataloaders, criterion=None, optimizer=None, scheduler=None, num_epochs=25):

        import time
        import copy
        from torch import set_grad_enabled, max, sum
        import torch.nn as nn
        import torch.optim as optim
        from torch.optim import lr_scheduler
        self.class_names = dataloaders['train'].dataset.dataset.dataset.classes
        [self.data_root.append(x) for x in dataloaders['train'].dataset.dataset.dataset.root if x not in self.data_root]
        device = self.device
        model = self.model
        model = model.to(device)

        if criterion == None:
            criterion = nn.CrossEntropyLoss()
        if optimizer == None:
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        if scheduler == None:
            scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, targets,_ in dataloaders[phase]:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = max(outputs, 1)
                        loss = criterion(outputs, targets)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += sum(preds == targets.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataloaders[phase].dataset.__len__()
                epoch_acc = running_corrects.double() / dataloaders[phase].dataset.__len__()

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        self._trained=True
        self.model=model



    def visualize_model(self, loader ,num_images=12, numcolumns = 4):
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
                    ax = plt.subplot(int(np.ceil(num_images/numcolumns)), numcolumns, images_so_far)
                    ax.axis('off')
                    try:
                        if labels[j] in class_names:
                            ax.set_title('predicted: {}'.format(class_names[preds[j]])+' ground truth: {}'.format(labels[j]))
                        else:
                            ax.set_title('predicted: {}'.format(class_names[preds[j]])+' ground truth: Unknown')
                    except:
                        ax.set_title('predicted: {}'.format(class_names[preds[j]])+' ground truth: Unknown')
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
        if layertype==None:
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
            if type(m)==layertype:
                # partial to assign the layer name to each hook
                handles.append(m.register_forward_hook(partial(save_activation, name)))
        labelsssss=[]
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
        for k,v in activations.items():
            print (k, v.size())

        # remove all hooks!
        [h.remove() for h in handles]
        return activations, labelsssss


    def get_logits_and_labels(self,dataloader_full):
        from torch import no_grad
        outcp=[]
        labelsssss=[]
        device = self.device
        model_ft = copy.deepcopy(self.model)
        #m = nn.Softmax(dim=1)
        with no_grad():
            for i, (inputs, targets,_) in enumerate(dataloader_full):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = (model_ft(inputs))
                outcp = outcp+([np.array(p) for p in outputs.cpu()])
                labelsssss = labelsssss + ([int(p) for p in targets.cpu()])
        outcp = np.array(outcp)
        labelsssss = np.array(labelsssss)
        return(outcp, labelsssss)

    def get_softmax_and_labels(self,dataloader_full):
        import torch.nn as nn
        from torch import no_grad
        m = nn.Softmax(dim=1)
        outcp=[]
        labelsssss=[]
        device = self.device
        model_ft = copy.deepcopy(self.model)
        with no_grad():
            for i, (inputs, targets,_) in enumerate(dataloader_full):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = m(model_ft(inputs))
                outcp = outcp+([np.array(p) for p in outputs.cpu()])
                labelsssss = labelsssss + ([int(p) for p in targets.cpu()])
        outcp = np.array(outcp)
        labelsssss = np.array(labelsssss)
        return(outcp, labelsssss)

    def get_predicts_and_labels(self,dataloader_full):
        import torch.nn as nn
        from torch import no_grad, max
        predicts=[]
        labelsssss=[]
        device = self.device
        model_ft = copy.deepcopy(self.model)
        with no_grad():
            for i, (inputs, targets,_) in enumerate(dataloader_full):
                inputs = inputs.to(device)
                targets = targets.to(device)
                _, preds = max(model_ft(inputs),1)
                predicts = predicts + ([int(p) for p in preds.cpu()])
                labelsssss = labelsssss + ([int(p) for p in targets.cpu()])
        predicts = np.array(predicts)
        labelsssss = np.array(labelsssss)
        return(predicts, labelsssss)

    def ConfusionMatrix(self, dataloader):
        from torchmetrics import ConfusionMatrix
        from mlxtend.plotting import plot_confusion_matrix
        from torch import tensor
        import matplotlib
        class_names = self.class_names
        cmat = ConfusionMatrix(num_classes=len(class_names))
        preds , labels = self.get_predicts_and_labels(dataloader)
        cmat(tensor(preds), tensor(labels))

        cmat_tensor = cmat.compute()
        cmat = cmat_tensor.numpy()

        fig, ax = plot_confusion_matrix(conf_mat=cmat, class_names=class_names, show_normed=True)


    def classification_report(self, dataloader):
        from torchmetrics import ConfusionMatrix
        from mlxtend.plotting import plot_confusion_matrix
        from torch import tensor
        import matplotlib
        class_names = self.class_names
        cmat = ConfusionMatrix(num_classes=len(class_names))
        preds , labels = self.get_predicts_and_labels(dataloader)

        from sklearn import metrics
        print(metrics.classification_report(labels,preds,target_names=class_names))


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

    if ax==None:
        ax = plt.gca()

    h = ax.get_children()

    Xlim = ax.get_xlim();
    Ylim = ax.get_ylim();
    for hi in h:
        try:
            offs = hi.get_offsets().data
            J = np.where((offs[:,0]>Xlim[0])*(offs[:,0]<Xlim[1])*(offs[:,1]>Ylim[0])*(offs[:,1]<Ylim[1]))
            J = J[0]
        except:
            continue
    return J
