import idx2numpy
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset

def binary(im):
    x = torch.zeros(1,28,28)
    y = torch.ones(1,28,28)
    return torch.where(im > 0.1, x, y)


def prepare_data():
    # read data
    t10k = idx2numpy.convert_from_file('data/t10k-images-idx3-ubyte')
    train_img = idx2numpy.convert_from_file('data/train-images-idx3-ubyte')
    train_lab = idx2numpy.convert_from_file('data/train-labels-idx1-ubyte')

    # define transforms
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=(-8, 8)),
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    # transform datasets
    new_train = []
    new_labels = []
    new_test = []

    for (x, y) in zip(train_img, train_lab):
        im = transform_test(Image.fromarray(np.uint8(x)))
        im = binary(im)
        new_train.append(im)
        new_labels.append(y)
        '''
        for j in range(4):
            im = transform_train(Image.fromarray(np.uint8(x)))
            new_train.append(im)
            new_labels.append(y)
        '''

    for x in t10k:
        im = transform_test(Image.fromarray(np.uint8(x)))
        im = binary(im)
        new_test.append(im)

    # convert to tensors of proper size
    train_tensor = torch.Tensor(len(new_train), 28, 28)
    torch.cat(new_train, out=train_tensor)
    new_train = train_tensor.view(-1, 1, 28, 28)

    test_tensor = torch.Tensor(len(new_test), 28, 28)
    torch.cat(new_test, out=test_tensor)
    new_test = test_tensor.view(-1, 1, 28, 28)

    new_labels = torch.tensor(new_labels)

    # create datasets
    train_data = TensorDataset(new_train, new_labels)
    test_data = TensorDataset(new_test)

    # creating samplers and data loaders
    batch_size = 10
    valid_size = 0.2
    num_train = len(new_train)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(num_train * valid_size))
    train_index, test_index = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(test_index)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler
    )
    valid_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=valid_sampler
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1
    )
    return train_loader, valid_loader, test_loader
