import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import pandas as pd

from py_src.prepare_data import prepare_data
from py_src.net import Net

print(f'main.py: Lets train !!!\n')
with open('train_log.txt', 'w') as f:
    print(f'main.py: Lets train !!!\n', file=f)

train_loader, valid_loader, test_loader = prepare_data()

model = Net()

# loss function (cross entropy loss)
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.SGD(model.parameters(), lr=0.05)

epochs = 30
batch = 12

max_accuracy = 0
torch.save(model.state_dict(), 'trained_model.pt')

for epoch in range(epochs):

    train_loss = 0
    valid_loss = 0
    train_correct = 0
    valid_correct = 0

    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        y = model(data)
        loss = criterion(y, target.long())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()*data.size(0)
        for (a, b) in zip (y, target):
            train_correct += (torch.argmax(a) == b).float()

    model.eval()
    for batch_index, (data, target) in enumerate(valid_loader):
        y = model(data)
        loss = criterion(y, target.long())
        valid_loss += loss.item() * data.size(0)
        for (a, b) in zip (y, target):
            valid_correct += (torch.argmax(a) == b).float()

    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)
    train_accuracy = 100 * train_correct / len(train_loader.sampler)
    valid_accuracy = 100 * valid_correct / len(valid_loader.sampler)

    print(f'Current Epoch: {epoch}\
    \nTraining Loss: {round(train_loss, 6)}\
    \nValidation Loss: {round(valid_loss, 6)}\
    \nTraining Accuracy: {train_accuracy}\
    \nValidation Accuracy: {valid_accuracy}')

    with open('train_log.txt', 'a') as f:
        print(f'Current Epoch: {epoch}\
        \nTraining Loss: {round(train_loss, 6)}\
        \nValidation Loss: {round(valid_loss, 6)}\
        \nTraining Accuracy: {train_accuracy}\
        \nValidation Accuracy: {valid_accuracy}', file=f)

    if max_accuracy < valid_accuracy:
        max_accuracy = valid_accuracy
        print("New Leader !!!\n")
        with open('train_log.txt', 'a') as f:
            print("New Leader !!!\n", file=f)
        torch.save(model.state_dict(), 'trained_model.pt')

# Make predictions
model.load_state_dict(torch.load('trained_model.pt'))

model.eval()
output = []
for batch_idx, (data,) in enumerate(test_loader):
    output.append([batch_idx, torch.argmax(model(data)).item()])

Submission = pd.DataFrame(output, columns=['Id', 'Class'])
Submission.to_csv('kaggle.out.csv', index=False)

print(Submission.head())
with open('train_log.txt', 'a') as f:
    print(Submission.head(), file=f)