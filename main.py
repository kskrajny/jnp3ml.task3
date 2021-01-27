import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np

from py_src.prepare_data import prepare_data
from py_src.net import Net

print("main.py: Let's train !!!")

train_loader, valid_loader, test_loader = prepare_data()

model = Net()

# loss function (cross entropy loss)
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.SGD(model.parameters(), lr=0.05)

epochs = 30
batch = 10

min_loss = np.inf

for epoch in range(epochs):

    train_loss = 0
    valid_loss = 0

    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        y = model(data)
        loss = criterion(y, target.long())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()*data.size(0)

    model.eval()
    for batch_index, (data, target) in enumerate(valid_loader):
        output = model(data)
        loss = criterion(output, target.long())
        valid_loss += loss.item() * data.size(0)

    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)

    print(f'Current Epoch: {epoch}\
    \nTraining Loss: {round(train_loss, 6)}\
    \nValidation Loss: {round(valid_loss, 6)}')

    if min_loss > valid_loss:
        print("New Leader !!!\n")
        torch.save(model.state_dict(), 'trained_model.pt')
