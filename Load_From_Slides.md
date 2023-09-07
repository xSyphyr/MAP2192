
``` shell 
%%capture
!apt-get install poppler-utils
!pip install pdf2image
```

```python
from pdf2image import convert_from_path
import requests
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize
```

``` python
def plot(x):
    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap = 'gray')
    ax.axis('off')
    fig.set_size_inches(5, 5)
    plt.show()

def get_google_slide(url):
    url_head = "https://docs.google.com/presentation/d/"
    url_body = url.split('/')[5]
    page_id = url.split('.')[-1]
    return url_head + url_body + "/export/pdf?id=" + url_body + "&pageid=" + page_id

def get_slides(url):
    url = get_google_slide(url)
    r = requests.get(url, allow_redirects=True)
    open('file.pdf', 'wb').write(r.content)
    images = convert_from_path('file.pdf', 500)
    return images
```

``` python
data_deck = "https://docs.google.com/presentation/d/1JuHDfsJL5S2unNAP_6iWqWC_wB_WmdSKYS6D0M1KvJ8/edit#slide=id.g206f8279a60_0_0"
```

``` python
image_list = get_slides(data_deck)
n = len(image_list)
```

``` python
for i in range(n):

    plot(image_list[i])
    print(np.array(image_list[i]).shape)
```

``` python
n = len(image_list)
h = 512
w = 512
c = 3
```

``` python
image_array = np.zeros((n,h,w,c))

for i in range(n):

    image = image_list[i]

    plot(image)
    image = np.array(image)
    print(image.shape)

    image = resize(image,(512,512))

    print(image.shape)

    image_array[i] = image
```
