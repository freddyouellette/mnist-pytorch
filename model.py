import torch.nn as nn
import torch.optim as optim

model_options = {
    "epochs": 3,
    "batch_size": 1000,
    "eval_batch_interval": 10,
    "learning_rate": 2e-3,
    "weight_decay": 0.01,
    "dropout": 0.3,
}

model = nn.Sequential(
    # input shape: 1 x 28 x 28 = 784
    
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
    # outputs 32 feature maps with the same size as the original image
    # shape: 32 x 28 x 28 = 25088
    
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    # outputs 32 feature maps with half the size of the original image
    # shape: 32 x 14 x 14 = 6272
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    # outputs 64 feature maps with the same size as the previous feature maps
    # shape: 64 x 14 x 14 = 12544
    
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    # outputs 64 feature maps with half the size of the previous feature maps
    # output size: 64 x 7 x 7 = 3136

    nn.Flatten(),

    nn.Linear(64 * 7 * 7, 1000),
    nn.ReLU(),
    nn.Dropout(p=model_options['dropout']),

    nn.Linear(1000, 128),
    nn.ReLU(),
    nn.Dropout(p=model_options['dropout']),
    nn.Linear(128, 10),
)

optimizer = optim.Adam(
    model.parameters(), 
    lr=model_options['learning_rate'],
    weight_decay=model_options['weight_decay'],
)

# scheduler will step at each eval
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60000 * model_options['epochs'] / model_options['batch_size'] / model_options['eval_batch_interval'])

loss_function = nn.CrossEntropyLoss()