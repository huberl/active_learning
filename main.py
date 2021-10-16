import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from time import time

import torch
import random
import numpy as np

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = 'cuda:0'


class ActiveLearningSampler:

    def __init__(self):
        self.model = None
        self.activations = None
        self.min_dists = None
        self.closest_centers = None
        self.activations = None
        self.acc_pos_centers = None
        self.acc_pos_unlabeled = None
        self.indices_center = None
        self.indices_unlabeled = None

    def update_model(self, model, loader_center, loader_unlabeled):
        '''
        Call this model after every full iteration (i.e. B runs) and after retraining the model with the new dataset.
        :param model:
        :param loader_center:
        :param loader_unlabeled:
        :return: None
        '''
        self.model = model
        center_activations = self._get_activations(loader_center)
        unlabeled_activations = self._get_activations(loader_unlabeled)
        self.activations = torch.cat((unlabeled_activations, center_activations))

        self.acc_pos_unlabeled = list(range(len(unlabeled_activations)))
        self.acc_pos_centers = list(
            range(len(unlabeled_activations), len(unlabeled_activations) + len(center_activations)))

        # Create set of indices to differentiate between the center activations and unlabeled activations
        self.indices_unlabeled = np.array(loader_unlabeled.dataset.indices)
        self.indices_center = np.array(loader_center.dataset.indices)

        assert not np.in1d(self.indices_unlabeled,
                           self.indices_center).any(), 'The two sets should be mutually exclusive'

        # Compute the initial min_dists and closest centers
        dists = self.pairwise_dist(self.activations[self.acc_pos_centers], self.activations[self.acc_pos_unlabeled])
        self.min_dists, self.closest_centers = dists.min(axis=0)

    def _get_activations(self, loader):
        '''
        Compute the activations for the penultimate layer for a given Dataloader.
        :param self:
        :param loader:
        :return:
        '''
        self.model.eval()

        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        self.model.fc.register_forward_hook(get_activation('fc'))

        activations = []
        for data in loader:
            imgs = data[0].to(device)

            with torch.no_grad():
                _ = self.model(
                    imgs)  # We don't care about the actual outputs. Only the activations of the penultimate layer

            activations.append(activation['fc'])

        return torch.cat(activations)

    def pairwise_dist(self, tnsr1, tnsr2, eps=1e-5):
        '''
        Computes the pairwise l2 distance between the two input tensors.
        :param tnsr1: First tensor where each row is one activation of a sample.
        :param tnsr2: First tensor where each row is one activation of a sample.
        :param eps:
        :return: The pairwise l2 distances.
        '''
        # https://github.com/josipd/torch-two-sample/blob/master/torch_two_sample/util.py
        n_1, n_2 = tnsr1.size(0), tnsr2.size(0)
        norms_1 = torch.sum(tnsr1 ** 2, dim=1, keepdim=True)
        norms_2 = torch.sum(tnsr2 ** 2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * tnsr1.mm(tnsr2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))

    def get_centers(self, B=128):
        '''
        :param new_center_index: Global index of the new center in the unlabeled dataset.
        :return:
        '''
        centers = []

        for _ in range(B):
            assert len(self.min_dists) == len(self.indices_unlabeled), 'min_dists and indices_unlabeled must have same ' \
                                                                       'length in order to get unlabeled sample with largest distance'
            new_center_index = self.indices_unlabeled[self.min_dists.argmax()]
            centers.append(new_center_index)

            # Add new center to list of centers if provided

            pos_in_ul = np.where(self.indices_unlabeled == new_center_index)[0][0]
            pos_in_acc = self.acc_pos_unlabeled.pop(pos_in_ul)
            self.acc_pos_centers = np.append(self.acc_pos_centers, pos_in_acc)

            self.indices_center = np.append(self.indices_center, new_center_index)
            self.indices_unlabeled = np.delete(self.indices_unlabeled,
                                               np.argwhere(self.indices_unlabeled == new_center_index))

            dists = self.pairwise_dist(self.activations[self.acc_pos_centers], self.activations[self.acc_pos_unlabeled])
            self.min_dists, min_pos = dists.min(axis=0)
            min_pos = list(min_pos.cpu().numpy())
            self.closest_center = self.indices_unlabeled[min_pos]  # Check which indices correspond to the positions

            assert not np.in1d(self.indices_unlabeled,
                               self.indices_center).any(), 'The two sets should be mutually exclusive'


        return centers, self.min_dists, self.closest_centers


def train_model(model, loader_centers, loader_val, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for _ in tqdm(range(num_epochs)):
        for data in loader_centers:
            imgs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    if loader_val is not None:
        correct = 0
        for data in loader_val:
            imgs, labels = data[0].to(device), data[1].to(device)

            with torch.no_grad():
                outputs = model(imgs)

            correct += sum(outputs.argmax(axis=1) == labels)

        print(f'\nAcc: {correct / len(loader_val.dataset):.2f}.    Used {len(loader_centers.dataset)} samples')


def get_model():
    torch.manual_seed(0)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=10, bias=True)
    torch.nn.init.xavier_uniform_(model.fc.weight)
    model.to(device)

    return model

def test(model):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='data/CIFAR10', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=8)

    total = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            pred = model(images)
            _, preds = torch.max(pred, 1)
            correct += (preds == labels).sum()
            total += len(labels)

    print(f'Test Accuracy: {correct / total:.2f}')


def baseline(num_samples):
    model = get_model().to(device)
    model.eval()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset_full = torchvision.datasets.CIFAR10(root='data/CIFAR10', train=True, download=True, transform=transform)
    trainset, _ = torch.utils.data.random_split(trainset_full, (
        num_samples, len(trainset_full) - num_samples))

    loader_train = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=False, num_workers=8)

    train_model(model, loader_train, None, num_epochs=10)
    test(model)


def run_active_learning():
    al_sampler = ActiveLearningSampler()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset_full = torchvision.datasets.CIFAR10(root='data/CIFAR10', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='data/CIFAR10', train=False, download=True, transform=transform)

    train_size = len(trainset_full)
    # Choose small, random set of training samples to begin with
    center_size, unlabeled_size, val_size = int(train_size * 0.02), int(train_size * 0.96), int(train_size * 0.02)
    trainset_centers, trainset_unlabeled, trainset_val = torch.utils.data.random_split(trainset_full, (
        center_size, unlabeled_size, val_size))

    print(f'Center size: {center_size}    Unlabeled size: {unlabeled_size}      Val size: {val_size}')

    loader_centers = torch.utils.data.DataLoader(trainset_centers, batch_size=512, shuffle=False, num_workers=8)
    loader_unlabeled = torch.utils.data.DataLoader(trainset_unlabeled, batch_size=512, shuffle=False, num_workers=8)
    loader_val = torch.utils.data.DataLoader(trainset_val, batch_size=512, shuffle=False, num_workers=8)

    iterations = 20
    cntrs_per_iter = 128
    model = get_model()

    for i in range(iterations):
        #model = get_model()
        train_model(model, loader_centers, loader_val, num_epochs=10)

        centers = trainset_centers.indices
        al_sampler.update_model(model, loader_centers, loader_unlabeled)

        max_before = al_sampler.min_dists.max()

        centers_new, min_dists, closest_centers = al_sampler.get_centers(cntrs_per_iter)
        centers.extend(centers_new)

        max_after = min_dists.max()
        print(f'Max distance:  Before {max_before:.2f}   After {max_after:.2f}')

        trainset_centers = torch.utils.data.Subset(trainset_full, centers)
        loader_centers = torch.utils.data.DataLoader(trainset_centers, batch_size=512, shuffle=False, num_workers=2)

        new_unlabeled_indices = list(set(loader_unlabeled.dataset.indices) - set(centers))
        trainset_unlabeled = torch.utils.data.Subset(trainset_full, new_unlabeled_indices)
        loader_unlabeled = torch.utils.data.DataLoader(trainset_unlabeled, batch_size=512, shuffle=False, num_workers=2)

    test(model)

baseline(3432)
run_active_learning()
