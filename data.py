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

def get_data(batch_size, log=True):
    transform_steps = transforms.Compose([
        transforms.RandomRotation(40, fill=0),
        transforms.RandomPerspective(fill=0),
        transforms.ElasticTransform(alpha=40.0),
        transforms.RandomAffine(0, (0.1,0.1), fill=0),
        transforms.RandomResizedCrop((28), scale=(1.0, 1.5)),
        transforms.RandomZoomOut(fill=0, side_range=(1.0, 2.0), p=1.0),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        AddRandomNoise(),
    ])

    train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_steps)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_steps)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)

    i = random.randint(0, len(test_data) - 1)
    random_sample_img, random_sample_label = test_data[i]

    if log:
        # show the number of rows and columns
        print( "# training samples:" + str(len(train_data)) )
        print( "# testing samples:" + str(len(test_data)) )
        # print( "cols (# features):" + str(random_sample_img.numel()) )
        print(f"image size: ({random_sample_img.shape[1]}, {random_sample_img.shape[2]})")
        print("features per sample:", random_sample_img.flatten().shape[0])
    
    return trainloader, testloader