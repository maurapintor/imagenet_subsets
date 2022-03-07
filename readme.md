# How to use this repository

1. Download the validation set of ImageNet.
2. Prepare the folder structure by using the file `valprep.sh`.
3. Use the class ImageNetSubset to load a custom set of classes.

Example:

```python
import torch.utils.data
from torchvision import transforms
from read_categories import ImageNetSubset

imagenet_dir = './data/IMAGENET/val'
transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
include = (0, 1, 10)
dataset_test = ImageNetSubset(imagenet_dir, transform=transform, include_list=include)

loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=16,
                                          shuffle=True, num_workers=1)

images, ls = next(iter(loader_test))

```

Printing the human-readable labels (taken from https://github.com/anishathalye/imagenet-simple-labels):

```python
import json

with open('imagenet-simple-labels.json', 'r') as f:
    labels = json.load(f)


def class_id_to_label(i):
    return labels[i]


print([class_id_to_label(l.item()) for l in ls])
```

Visualization of the samples (from the torchvision tutorials):

```python
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import make_grid


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


grid = make_grid(images)
show(grid)
plt.show()
```

