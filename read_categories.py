import numpy as np
import torchvision

class ImageNetSubset(torchvision.datasets.ImageFolder):
    def __init__(self, *args, include_list=tuple(range(10)),
                 nb_samples=None, reset_index=False, **kwargs):
        """
        Extracts a subset of the ImageNet dataset, including only
        selected classes and a specified number of samples.

        :param args:
        :param include_list: list containing the indexes of the classes
            from the ImageNet original classes that should be included in
            the subset.
        :param nb_samples: if specified, returns a dataset with the selected
            number of samples, taken randomly from the subset.
        :param reset_index: if specified, it resets the labels of the
            extracted subset to map to a range from 1 to len(include_list).
        """
        super(ImageNetSubset, self).__init__(*args, **kwargs)

        if include_list == []:
            raise ValueError("Parameter 'include_list' is an empty list. Select at least one class.")

        labels = np.array(self.targets)
        include = np.array(include_list).reshape(1, -1)
        mask = np.where((labels.reshape(-1, 1) == include).any(axis=1))[0]


        if nb_samples is not None:
            # select a subset of the samples
            indices = list(range(len(mask)))
            np.random.shuffle(indices)
            indices = indices[:nb_samples]
            mask = [mask[i] for i in indices]

        self.imgs = [self.imgs[i] for i in mask]
        self.samples = [self.samples[i] for i in mask]
        self.targets = [self.targets[i] for i in mask]

        if reset_index:
            # re-enumerate the labels so that they are a range starting from
            # zero and stopping at len(include_list)
            for new_idx, cl in enumerate(include_list):
                self.targets[self.targets == cl] = new_idx

        self.classes = [self.classes[i] for i in include_list]
