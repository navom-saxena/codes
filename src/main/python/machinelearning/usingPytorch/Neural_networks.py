import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

root = '/Users/navomsaxena/Downloads'
dataset = MNIST(root=root, download=False, transform=ToTensor())

val_size = 10000
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])
len(train_ds), len(val_ds)

batch_size = 128

train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size * 2, num_workers=0, pin_memory=True)


def accuracy(out, labels):
    _, p = torch.max(out, dim=1)
    return torch.tensor(torch.sum(p == labels).item() / len(p))


def epoch_end(epoch, result):
    print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


def validation_epoch_end(outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}


def evaluate(m, vl):
    outputs = [m.validation_step(batch) for batch in vl]
    return validation_epoch_end(outputs)


class MnistModel(nn.Module):
    def __init__(self, input_size, hidden_layer, output_layer):
        super(MnistModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_layer)
        self.linear2 = nn.Linear(hidden_layer, output_layer)

    def forward(self, xb):
        xb = xb.view(xb.size(0), -1)
        hidden_out = self.linear1(xb)
        out_relu = F.relu(hidden_out)
        out = self.linear2(out_relu)
        return out

    def training_step(self, batch):
        images, label = batch
        out = self(images)
        loss = F.cross_entropy(out, label)
        return loss

    def validation_step(self, batch):
        images, label = batch
        out = self(images)
        loss = F.cross_entropy(out, label)
        acc = accuracy(out, label)  # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}


ip_size = 784
hidden_size = 32  # you can change this
num_classes = 10
model = MnistModel(ip_size, hidden_layer=hidden_size, output_layer=num_classes)

for t in model.parameters():
    print(t.shape)


def fit(m, epoch, lr, train_dl, val_dl, opt_func=torch.optim.SGD):
    h = []
    optimiser = opt_func(m.parameters(), lr=lr)
    for e in range(epoch):
        for batch in train_dl:
            loss = m.training_step(batch=batch)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
        res = evaluate(m, val_dl)
        epoch_end(e, result=res)
        h.append(res)

    return h


history = [evaluate(model, val_loader)]
print(history)
history += fit(model, 50, 0.1, train_loader, val_loader)
