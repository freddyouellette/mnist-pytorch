import torch
import torchvision
import torchvision.transforms.v2 as transforms
import random
import numpy as np

# prepare preprocessing of data
class AddRandomNoise(object):
    def __init__(self, mean=0.0, std1=0.1, std2=0.2):
        self.mean = mean
        self.std1 = std1  # Lower bound of the standard deviation range
        self.std2 = std2  # Upper bound of the standard deviation range

    def __call__(self, tensor):
        # U-shaped distribution for the standard deviation of the noise
        if random.random() < 0.5:
            actual_std = np.abs(np.random.normal(self.std1, self.std1 / 2))
        else:
            actual_std = np.abs(np.random.normal(self.std2, self.std2 / 2))
        
        # Clip the standard deviation to be within 0 and the maximum desired standard deviation
        actual_std = np.clip(actual_std, 0, self.std2)

        noise = torch.randn(tensor.size()) * actual_std + self.mean
        return tensor + noise

def get_transforms(sample_augmentation_ratio=0.5):
    return transforms.Compose([
        # only apply transformations to x% of data, to prevent learning ONLY on transformed images
        transforms.RandomChoice([
            transforms.Compose([
                transforms.RandomRotation(40, fill=0),
                transforms.RandomPerspective(fill=0),
                transforms.ElasticTransform(alpha=40.0),
                transforms.RandomAffine(0, (0.1,0.1), fill=0),
                transforms.RandomResizedCrop((28), scale=(1.0, 1.5)),
                transforms.RandomZoomOut(fill=0, side_range=(1.0, 2.0), p=1.0),
                transforms.Resize((28, 28)),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize((0.5,), (0.5,)),
                AddRandomNoise(),
            ]),
            transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize((0.5,), (0.5,)),
            ]),
        ], p=[sample_augmentation_ratio, 1 - sample_augmentation_ratio]),
    ])

def get_training_data(batch_size, sample_augmentation_ratio=0.5):
    train_transform_steps = get_transforms(sample_augmentation_ratio)
    train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform_steps)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    return trainloader

def get_testing_data(batch_size, sample_augmentation_ratio=0.5):
    test_transform_steps = get_transforms(sample_augmentation_ratio)
    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transform_steps)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)
    return testloader