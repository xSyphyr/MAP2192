# Problem Set 1
## [Colab Notebook](https://colab.research.google.com/drive/1uopl_1hSQJcePGaWwynvVacFIRyKJQ4o?usp=sharing)

## Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from skimage.util import montage
from skimage.io import imread

## Define a function to move data to the GPU
``` python
def GPU(data):
    return torch.tensor(data, requires_grad=True, dtype=torch.float, device=torch.device('cuda'))
```
## Define another function to move data to the GPU without gradients
``` python
def GPU_data(data):
    return torch.tensor(data, requires_grad=False, dtype=torch.float, device=torch.device('cuda'))
```
## Define a function to plot an image
```python
def plot(x):
    if type(x) == torch.Tensor:
        x = x.cpu().detach().numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap='gray')
    ax.axis('off')
    fig.set_size_inches(7, 7)
    plt.show()
```
## Define a function to create a montage and plot it
``` python
def montage_plot(x):
    x = np.pad(x, pad_width=((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    plot(montage(x))
```
## Load the MNIST training and testing datasets
``` python
train_set = datasets.MNIST('./data', train=True, download=True)
test_set = datasets.MNIST('./data', train=False, download=True)
```
## Extract data and labels from the training and testing sets
``` python
X = train_set.data.numpy()
X_test = test_set.data.numpy()
Y = train_set.targets.numpy()
Y_test = test_set.targets.numpy()
```
## Normalize the data and add a channel dimension
``` python
X = X[:, None, :, :] / 255
X_test = X_test[:, None, :, :] / 255
```
## Reshape the data
``` python
X = X.reshape(X.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)
```
## Move the data to the GPU
``` python
X = GPU_data(X)
Y = GPU_data(Y)
X_test = GPU_data(X_test)
Y_test = GPU_data(Y_test)
```
## Transpose the data
``` python
X = X.T
```
## Select a single column (sample) from the data
``` python
x = X[:, 0:1]
```
## Initialize a random matrix and perform a matrix multiplication
``` python
M = GPU(np.random.rand(10, 784))
y = M @ x
batch_size = 64
```
## Select a batch of columns (samples) from the data
``` python
x = X[:, 0:batch_size]
```
## Initialize a new random matrix and perform a matrix multiplication
``` python
M = GPU(np.random.rand(10, 784))
y = M @ x
```
## Find the index with the highest value for each column
``` python
y = torch.argmax(y, 0)
```
## Calculate the accuracy for the batch
``` python
accuracy = torch.sum((y == Y[0:batch_size])) / batch_size
```
## Initialize variables to track the best accuracy and corresponding matrix
``` python
m_best = 0
acc_best = 0
```
## Perform an optimization loop
``` python
for i in range(100000):
    step = 0.0000000001

    # Generate a random matrix
    m_random = GPU_data(np.random.randn(10, 784))

    # Update the matrix
    m = m_best + step * m_random

    # Perform a matrix multiplication
    y = m @ X

    # Find the index with the highest value for each column
    y = torch.argmax(y, axis=0)

    # Calculate the accuracy
    acc = ((y == Y)).sum() / len(Y)

    # Check if the current accuracy is better than the best
    if acc > acc_best:
        print(acc.item())
        m_best = m
        acc_best = acc
```
