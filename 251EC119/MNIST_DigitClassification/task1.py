import torch
import torchvision
import matplotlib.pyplot as plt

#loading dataset
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())

#to print shape
image, label = dataset[0]#first of the dataset
print(f"Image Shape: {image.shape}") # 28 by 28 in grayscale

# diaplaying  samples
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i in range(5):
    img, lbl = dataset[i]
    axes[i].imshow(img.squeeze(), cmap='gray')
    axes[i].set_title(f"Label: {lbl}")
    axes[i].axis('off')
plt.show()