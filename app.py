import torch
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from torchvision import transforms
import torch.nn as nn

device_name = "cuda" if torch.cuda.is_available() else "cpu"
print("Using GPU with CUDA" if device_name == "cuda" else "Using CPU")
device = torch.device(device_name)

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
    nn.Dropout(p=0.4),

    nn.Linear(1000, 128),
    nn.ReLU(),
    nn.Dropout(p=0.4),
    nn.Linear(128, 10),
)
model.load_state_dict(torch.load('model.pth'))
model.eval()
model.to(device)

transform_steps = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

SIZE = 192
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=6,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw",
    point_display_radius=0,
    key='canvas')

img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
# st.write('Model Input')
# st.image(rescaled)
test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
pil_image = Image.fromarray(test_x)
input_tensor = transform_steps(pil_image)
input_tensor = input_tensor.unsqueeze(0)
input_tensor = input_tensor.to(device)

with torch.no_grad():
    outputs = model(input_tensor)

_, predictions = torch.max(outputs, 1)
st.write(f'MODEL PREDICTION: {predictions[0]}')