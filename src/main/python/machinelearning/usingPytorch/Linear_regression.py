import torch
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


# https://jovian.ai/aakashns/02-linear-regression


def model(x):
    return x @ w.t() + b


def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()


inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype='float32')

targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)

prediction = None
for i in range(500):
    prediction = model(inputs)
    loss = mse(prediction, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()

# print(prediction)
# print(targets)

inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70],
                   [74, 66, 43],
                   [91, 87, 65],
                   [88, 134, 59],
                   [101, 44, 37],
                   [68, 96, 71],
                   [73, 66, 44],
                   [92, 87, 64],
                   [87, 135, 57],
                   [103, 43, 36],
                   [68, 97, 70]],
                  dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119],
                    [57, 69],
                    [80, 102],
                    [118, 132],
                    [21, 38],
                    [104, 118],
                    [57, 69],
                    [82, 100],
                    [118, 134],
                    [20, 38],
                    [102, 120]],
                   dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
train_ds = TensorDataset(inputs, targets)

batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

model = nn.Linear(3, 2)
print(model.weight)
print(model.bias)

preds = model(inputs)
loss_fn = F.mse_loss
opt = torch.optim.SGD(model.parameters(), lr=1e-5)


def train(epochs, m, l_fn, grad, dl):
    for epoch in range(epochs):
        for xb, yb in dl:
            pred = m(xb)
            loss_v = l_fn(pred, yb)
            loss_v.backward()
            grad.step()
            grad.zero_grad()

        if epoch % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))


train(300, model, loss_fn, opt, train_dl)

new_pred = model(inputs)
print(new_pred)
print(targets)

print(model(torch.tensor([[75, 63, 44.]])))
