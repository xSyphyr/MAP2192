[Colab Notebook](https://colab.research.google.com/drive/1uEmvfhurlNPLU-RBffxTx9xlf0r3H-j7?usp=sharing)

# Code Documentation

This Markdown document provides documentation for the provided Python code, which involves various image processing and convolution operations using libraries such as NumPy, Matplotlib, PyTorch, scikit-image, and more.

## Imports

```python
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from scipy.signal import convolve2d
from skimage import data, color, io
import IPython
```

The code begins by importing necessary Python libraries and dependencies for image processing and visualization.

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from skimage.util import montage
!pip install wandb
import wandb as wb
from skimage.io import imread

import imageio as io

import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

from scipy import signal
```

In this part of the code, additional libraries are imported to support image processing, deep learning, and data visualization.

## Image Loading and Preprocessing

```python
# Define a function for plotting images
def plot(x):
    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap='gray')
    ax.axis('off')
    fig.set_size_inches(5, 5)
    plt.show()
```

A function `plot` is defined for displaying grayscale images without axis labels.

```python
# Load and process an image from a URL
image = io.imread("http://harborparkgarage.com/img/venues/pix/aquarium2.jpg")
image = image[:, :, :]
image = image.astype float()
image /= 255
plot(image)
```

This section loads an image from a URL, converts it to grayscale, and displays it using the `plot` function.

## Image Convolution and Filtering

```python
# Create random filters and perform convolution
filters = np.random.random((96, 11, 11, 3))
image = np.transpose(image, (2, 0, 1))
f = np.random.random((1, 3, 11, 11))
image = image[None, :, :, :]
f = torch.from_numpy(f)
image = torch.from_numpy(image)
image2 = F.conv2d(image, f)
image2 = image2.numpy()
image2[0, 0, :, :].shape
plot(image2[0, 0, :, :])
```

In this part, random filters are created, and convolution is performed on the image with these filters. The result is displayed using the `plot` function.

```python
x = image2
```

The variable `x` is assigned the value of `image2` for further processing.

## Custom Convolution Function

```python
# Define a custom convolution function
def conv2(x, f):
    x2 = np.zeros(x.shape)
    for i in range(1, x.shape[0]-1):
        for j in range(1, x.shape[1]-1):
            x2[i, j] = f[0, 0] * x[i-1, j-1] + f[0, 1] * x[i-1, j] + f[0, 2] * x[i-1, j+1] + f[1, 0] * x[i, j-1] + f[1, 1] * x[i, j] + f[1, 2] * x[i, j+1] + f[2, 0] * x[i+1, j-1] + f[2, 1] * x[i+1, j] + f[2, 2] * x[i+1, j+1]
    return x2
```

The `conv2` function is defined to perform custom 2D convolution on an input image.

```python
a = np.matrix([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1])
z = conv2(x, a)
x = x[0, 0, :, :]
plot(x)
```

A custom convolution filter `a` is applied to `x` using the `conv2` function, and the result is displayed.

## Applying Random Convolution Filters

```python
# Apply random convolution filters to the image
for i in range(10):
    a = 2 * np.random.random((3, 3)) - 1
    print(a)
    z = conv2(x, a)
    plot(z)
```

Random convolution filters are applied to the image in a loop, and the results are displayed.

This code provides a demonstration of image processing techniques, including convolution and filtering operations, using various libraries for image handling and visualization.
