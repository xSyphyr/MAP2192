``` python
import numpy as np
import matplotlib.pyplot as plt

# Define the dimensions of the image (e.g., 256x256 pixels)
width, height = 256, 256

# Generate a random grayscale image
random_image = np.random.randint(0, 256, (height, width), dtype=np.uint8)

# Display the image using Matplotlib
plt.imshow(random_image, cmap='gray')
plt.title('Random Grayscale Image')
plt.axis('off')  # Turn off axis labels and ticks
plt.show()
```
