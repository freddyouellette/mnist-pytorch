import torch
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from torchvision import transforms
from model import model
import os

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

if not os.path.exists('model.pth'):
    st.write('Model not found. Please train the model first with the train.ipynb Jupyter Notebook.')
    exit()

model.load_state_dict(torch.load('model.pth'))
model.eval()
model.to(device)

transform_steps = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

st.markdown('# Handwritten Digit Recognition')
st.write('Draw a digit from 0 to 9 in the canvas below to see the model\'s prediction.')

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

result = st.empty()

result.markdown(f'# Model Prediction: ...')

with torch.no_grad():
    outputs = model(input_tensor)

_, predictions = torch.max(outputs, 1)
result.markdown(f'# Model Prediction: {predictions[0]}')