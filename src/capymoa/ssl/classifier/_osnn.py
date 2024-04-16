"""
R. G. F. Soares and L. L. Minku, “Osnn: An online semisupervised
neural network for nonstationary data streams,” IEEE Transactions
on Neural Networks and Learning Systems, pp. 1–13, 2021.

CapyMOA implementation by Botao, Anton

"""

import numpy as np
import random
import torch.nn as nn
import torch
from scipy.spatial.distance import cdist

from capymoa.base import ClassifierSSL


def kernel_fun(a, b, sigma):
    A = torch.sum((a - b) ** 2, dim=1)
    B = A / (2 * sigma**2)
    C = torch.exp(-B)
    return C


def Euclidean_Distances(a, b):
    dis = torch.sqrt(torch.sum((a - b) ** 2, dim=1))
    return dis


class OSNeuralNetwork(nn.Module):
    def __init__(self, num_center, n_out, window_size, beta=1, gamma=1):
        super(OSNeuralNetwork, self).__init__()
        self.n_out = n_out
        self.num_centers = num_center

        self.window_size = window_size
        self.data_window = torch.zeros(window_size, 1)
        self.label_window = torch.zeros(window_size, 1)
        self.plabel_window = torch.zeros(window_size, 1)

        self.i = 0
        self.beta = beta
        self.gamma = gamma

    def kernel_fun(self, batches):
        n_input = batches.size(0)
        A = self.centers.view(self.num_centers, -1).repeat(n_input, 1, 1)
        B = batches.view(n_input, -1).unsqueeze(1).repeat(1, self.num_centers, 1)
        C = torch.exp(-self.sigma.mul((A - B).pow(2).sum(2, keepdim=False)))
        return C

    def forward(self, batches):
        radial_val = self.kernel_fun(batches)
        class_score = self.linear(torch.cat([batches, radial_val], dim=1))
        return class_score

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def update_sigma(self):
        # The width of basis function is set to a proportion β of the mean of the Euclidean distances to the other centers.
        self.sigma = torch.ones(1, self.num_centers)
        for i in range(self.num_centers):
            dis = Euclidean_Distances(self.centers[i], self.centers)
            dis = torch.sum(dis) / (self.num_centers)
            self.sigma[0][i] = dis * self.beta

    def window_update(self, data, label):
        # The window is updated according to random sampling, and the first-in-first-out principle is adopted.
        if self.i == 0:
            self.data_window = torch.zeros(
                [self.window_size, data.size(1)], dtype=torch.float32
            )
            self.label_window = torch.zeros(
                [self.window_size, self.n_out], dtype=torch.float32
            )
            self.label_index = torch.zeros((self.window_size, 1), dtype=torch.float32)

            self.linear = nn.Sequential(
                nn.Linear(self.num_centers + data.size(1), self.n_out, bias=True),
                nn.Sigmoid(),
            )

        for i in range(data.size(0)):
            self.data_window = torch.cat(
                [self.data_window[1:, :], data[i : i + 1, :]], dim=0
            )
            self.label_window = torch.cat(
                [self.label_window[1:, :], label[i : i + 1, :]], dim=0
            )

            if label[i] != -1:
                self.label_index = torch.cat(
                    [self.label_index[1:, :], torch.ones(1, 1)], dim=0
                )
            else:
                self.label_index = torch.cat(
                    [self.label_index[1:, :], torch.zeros(1, 1)], dim=0
                )

            self.i = self.i + 1

            if self.i == self.window_size:
                index = torch.LongTensor(
                    random.sample(range(self.data_window.size(0)), self.num_centers)
                )
                self.centers = torch.index_select(self.data_window, 0, index)
                self.initialize_weights()

        if self.i % self.window_size == 0:
            self.center_adjustment()
            self.update_sigma()
            self.pseudo_label()
            update = True
        else:
            update = False
        return update

    def center_adjustment(self):
        # The samples are assigned to the nearest RBF centers, and then each center is updated according to the assigned samples.
        distances = np.linalg.norm(
            self.data_window[:, np.newaxis] - self.centers, axis=2
        )
        nearest_centers = np.argmin(distances, axis=1)
        assigned_samples = [
            self.data_window[nearest_centers == i] for i in range(len(self.centers))
        ]
        assigned_labels = [
            self.label_window[nearest_centers == i] for i in range(len(self.centers))
        ]
        assigned_label_index = [
            self.label_index[nearest_centers == i] for i in range(len(self.centers))
        ]

        for i in range(self.num_centers):
            if len(assigned_samples) > 0:
                unlabel_index = torch.squeeze(assigned_label_index[i] == 0.0, 1)
                label_index = torch.squeeze(assigned_label_index[i] == 1.0, 1)

                unlabel_sample = assigned_samples[i][unlabel_index]
                label_sample = assigned_samples[i][label_index]
                labels = assigned_labels[i][label_index]

                if len(label_sample) == 0 and len(unlabel_sample > 0):
                    self.centers[i] = torch.mean(unlabel_sample, axis=0)
                elif len(label_sample) > 0 and len(unlabel_sample > 0):
                    unique, counts = np.unique(labels, return_counts=True)
                    majorit_class = unique[np.argmax(counts)]
                    minorit_class = unique[np.argmin(counts)]
                    if majorit_class == minorit_class:
                        self.centers[i] = (
                            torch.mean(unlabel_sample, axis=0)
                            + torch.mean(label_sample, axis=0)
                        ) / 2
                    else:
                        majorit_sample = label_sample[labels.flatten() == majorit_class]
                        minorit_sample = label_sample[labels.flatten() == minorit_class]
                        a = (
                            majorit_sample.sum(dim=0) + minorit_sample.sum(dim=0)
                        ) / len(label_sample)
                        b = torch.mean(unlabel_sample, axis=0)
                        c = (
                            (len(majorit_sample) - len(minorit_sample))
                            / len(label_sample)
                        ) + 1
                        self.centers[i] = (a + b) / c
                elif len(label_sample) > 0 and len(unlabel_sample == 0):
                    unique, counts = np.unique(labels, return_counts=True)
                    majorit_class = unique[np.argmax(counts)]
                    minorit_class = unique[np.argmin(counts)]
                    majorit_sample = label_sample[labels.flatten() == majorit_class]
                    minorit_sample = label_sample[labels.flatten() == minorit_class]
                    a = (majorit_sample.sum(dim=0) + minorit_sample.sum(dim=0)) / len(
                        label_sample
                    )
                    c = (len(majorit_sample) - len(minorit_sample)) / len(label_sample)
                    self.centers[i] = a / c
            else:
                self.centers[i] = self.data_window[
                    torch.randint(self.data_window.shape[0], size=(1,))
                ][0]

        self.update_sigma()

    def pseudo_label(self):
        # Pseudo-labels for unlabeled samples are calculated based on the true labels of labeled samples and the output of the network on unlabeled samples.
        V = torch.cat([self.data_window, self.centers], dim=0)
        label = np.vstack((self.label_window, np.zeros((self.num_centers, 1))))
        label_index = np.vstack((self.label_index, np.zeros((self.num_centers, 1))))

        pre = self.forward(V)

        distances = cdist(V, V)
        nearest_distances = np.sort(distances, axis=1)[:, 1]
        nearest_distances = self.gamma * nearest_distances.reshape(-1, 1)

        S = np.exp(-1 * np.square(distances) / (nearest_distances + 1e-8))
        y = np.where(label_index, label, pre.detach().numpy())
        U = np.dot(S, y) / np.sum(S, axis=1).reshape(-1, 1)

        U = np.where(label_index, label, U)

        self.plabel_window = torch.from_numpy(U[: len(U) - self.num_centers])

    def return_window(self):
        # Returns the samples, pseudo-labels and true labels within the windows.
        return self.data_window, self.plabel_window, self.label_index


class def_loss(nn.Module):
    def __init__(self, model, lam=0.3, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.model = model
        self.lam = lam

    def L2loss(self):
        # l2 regularization on the network weights.
        l2_loss = torch.tensor(0.0, requires_grad=True)
        for name, parma in self.model.named_parameters():
            if "bias" not in name:
                l2_loss = l2_loss + (0.5 * torch.sum(torch.pow(parma, 2)))
        return l2_loss

    def forward(self, y_pred, y_true, label_index):
        # Computes supervised loss for labeled samples and unsupervised loss for unlabeled samples.
        labeled = torch.squeeze(label_index == 1.0, 1)
        unlabeled = torch.squeeze(label_index == 0.0, 1)

        y_pred_labeled = y_pred[labeled]
        y_true_label = y_true[labeled]

        y_pred_unlabeled = y_pred[unlabeled]
        y_sudo_unlabeled = y_true[unlabeled]

        first_item = -torch.mean(
            y_true_label * torch.log(y_pred_labeled + 1e-8)
            + (1 - y_true_label) * torch.log(1 - y_pred_labeled + 1e-8)
        )
        second_item = -torch.mean(
            y_sudo_unlabeled * torch.log(y_pred_unlabeled + 1e-8)
            + (1 - y_sudo_unlabeled) * torch.log(1 - y_pred_unlabeled + 1e-8)
        )
        l2_loss = self.L2loss() / len(y_pred)

        loss = first_item + self.lam * second_item + self.alpha * l2_loss
        return loss


class OSNN(ClassifierSSL):
    def __init__(
        self,
        schema=None,
        num_center=10,
        n_out=1,
        window_size=200,
        beta=1,
        gamma=1,
        optim_steps=200,
        seed=1,
    ):
        self.Network = OSNeuralNetwork(
            num_center=num_center,
            n_out=n_out,
            window_size=window_size,
            beta=beta,
            gamma=gamma,
        )

        self.window_size = window_size
        self.optim_steps = optim_steps
        self.loss_f = def_loss(model=self.Network)

        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.i = -1

    def train(self, instance):
        self._train(instance.x, instance.y_index)

    def train_on_unlabeled(self, instance):
        self._train(instance.x, -1)

    def _train(self, x, y):
        data = torch.tensor(np.array([x]), dtype=torch.float32)
        label = torch.unsqueeze(torch.tensor(np.array([y]), dtype=torch.float32), 1)

        update = self.Network.window_update(data, label)

        if update:
            self.i = 1
            params = self.Network.parameters()
            optimizer = torch.optim.LBFGS(params, lr=1e-3)
            for _ in range(self.optim_steps):
                data_window, label_window, label_index = self.Network.return_window()

                def closure():
                    optimizer.zero_grad()
                    y = self.Network.forward(data_window)
                    loss = self.loss_f(y, label_window, label_index)
                    loss.backward()
                    return loss

                self.Network.pseudo_label()
                optimizer.step(closure)

    def predict_proba(self, instance):
        if self.i == -1:
            return None
        X = torch.tensor(instance.x, dtype=torch.float32).reshape(1, -1)
        proba = self.Network.forward(X)
        proba = torch.cat((1 - proba, proba), dim=1).detach().numpy()
        return proba

    def predict(self, instance):
        if self.i == -1:
            return None
        return np.argmax(self.predict_proba(instance))

    def __str__(self):
        return f"OSNN(num_center={self.Network.num_centers}, window_size={self.window_size}, beta={self.Network.beta}, gamma={self.Network.gamma})"
