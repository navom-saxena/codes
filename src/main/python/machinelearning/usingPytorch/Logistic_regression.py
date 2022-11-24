import torch
from torch import Tensor
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

root = '/Users/navomsaxena/Downloads'
raw_dataset = MNIST(root=root, download=False)
test_dataset = MNIST(root=root, download=False, train=False)

# print(len(test_dataset))

dataset = MNIST(root=root, train=True, transform=transforms.ToTensor())
img_tensor, label = dataset[0]
# print(img_tensor.shape, label)
#
# print(img_tensor[0, 10:15, 10:15])
# print(torch.max(img_tensor), torch.min(img_tensor))

train_ds, val_ds = random_split(dataset, [50000, 10000])
len(train_ds), len(val_ds)

batch_size = 128

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

input_size = 28 * 28
num_classes = 10

# Logistic regression model
model = nn


# print(model.weight.shape)


def epoch_end(epoch, result):
    print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


def validation_epoch_end(output):
    batch_losses = [x['val_loss'] for x in output]
    epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
    batch_accs = [x['val_acc'] for x in output]
    epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb: Tensor) -> Tensor:
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out

    def training_step(self, batch):
        img, l = batch
        out = self(img)
        loss = F.cross_entropy(out, l)
        return loss

    def validation_step(self, batch):
        img, l = batch
        out = self(img)
        loss = F.cross_entropy(out, l)
        acc = accuracy(out, l)
        return {'val_loss': loss, 'val_acc': acc}


model = MnistModel()
# print(model.linear)

for images, labels in train_loader:
    # print(images.shape)
    outputs = model(images)
    # print(outputs)
    # print('outputs.shape : ', outputs.shape)
    # print('Sample outputs :\n', outputs[:2].data)
    probs = F.softmax(outputs, dim=1)
    # print(probs)
    break


def accuracy(output, l):
    _, p = torch.max(output, dim=1)
    return torch.tensor(torch.sum(p == l).item() / len(p))


def evaluate(m, v_s):
    op = [m.validation_step(batch) for batch in v_s]
    return validation_epoch_end(op)


def fit(epoch, lr, m, t_set, v_set, opt_func=torch.optim.SGD):
    optimiser = opt_func(model.parameters(), lr)
    history = []
    for e in range(epoch):
        for batch in t_set:
            loss = m.training_step(batch)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

        result = evaluate(m, v_set)
        epoch_end(e, result)
        history.append(result)

    return history


(fit(1, 0.01, model, train_loader, val_loader))
