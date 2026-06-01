import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# training dataset and test dataset
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# creating DataLoaders for train and test
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)#shuffle is set to True to randiomise the training set so that model does not just memorise the training set

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False
)#test dataset has shuffle set to False to get a fair evaluation of the model

# to print shape of a single batch
if __name__ == "__main__":
    #first batch from the training loader
    images, labels = next(iter(train_loader))

    print("checkpoint")
    print(f"images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Expected image shape: torch.Size([64, 1, 28, 28])")