"""
Implement convolutional neural net classifier for a tassk of 28 by 28 pixel images
of letters, will not have preprocessing dimensionality reduction
"""
from emnist import GetEMNIST
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

class ConvNet(nn.Module):
    # number of classes
    k = 9

    def __init__(self):
        super(ConvNet, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)


        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected, final layer holds k neurons
        self.fc1 = nn.Linear(32 * 5 * 5, self.k)

    def forward(self, x):
        x = x.unsqueeze(1).float()
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        # Flatten
        out = out.view(out.size(0), -1)
        # Dense
        out = self.fc1(out)
        return out


# Cap observations
n = 20000
k = 9
E_img = GetEMNIST( ['a', 'b', 'Cc', 'd', 'e', 'A', 'B', 'D', 'E'], n)
# Remeber to update k ^^
#E_img = GetEMNIST( ['T', 't', 'N', 'n'], n)
model = ConvNet()

# Integer mappings
#D = {"T":0, "t":1, "N":2, "n":3}
D = {'a':0, 'b':1, 'Cc':2, 'd':3, 'e':4, 'A':5, 'B':6, 'D':7, 'E':8}
int_labels = torch.tensor([D[i] for i in E_img.labels])

# Convert to torch dataset
all = TensorDataset(torch.tensor(E_img.X), int_labels)
train, test = random_split(all, lengths=[14000, 6000])

# Main train block
batch_size = 128
params = {
    'batch_size': batch_size,
    'shuffle': True,
}
loader_tr = DataLoader(train, **params)
loader_te = DataLoader(test, **params)

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_criteria = torch.nn.CrossEntropyLoss()

if torch.cuda.is_available():
    model = model.to("cuda")
    loss_criteria = loss_criteria.to("cuda")

for epoch in range(30):
    # Store losses and accuracy on each batch
    loss_list = []
    correct = 0
    total = 0
    # Train
    model.train()
    for batch_idx, batch in enumerate(loader_tr):
        #print("\nBatch = " + str(batch_idx))
        X, y = batch
        X = X.to("cuda")
        y = y.to("cuda")
        optimizer.zero_grad()

        out = model(X)
        predicted = torch.max(out.data, 1)[1]
        loss = loss_criteria(out, y)
        loss_list.append(loss.item())
        correct += (predicted == y).sum()
        total += len(y)
        # Parameters update, the pytorch strength here
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: t-loss = {np.mean(loss_list):.4f}, t-acc = {correct/total:.4f}")

# Eval
model.eval()
running_correct = 0
for batch_idx, batch in enumerate(loader_te):
    X, y = batch
    X = X.to("cuda")
    y = y.to("cuda")
    out = model(X)
    softed = torch.softmax(out, dim=1)
    running_correct += float(torch.sum(torch.argmax(softed, axis=1) == y))

print(f"Test Accuracy: {running_correct / 6000:.4f}")
