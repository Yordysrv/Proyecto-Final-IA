import gradio as gr
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import requests
import openai
import json

# CONFIGURA TU API KEY DE SPOONACULAR Y OPENAI
SPOONACULAR_API_KEY = "TU_API_KEY_SPOONACULAR"
OPENAI_API_KEY = "TU_API_KEY_OPENAI"
openai.api_key = OPENAI_API_KEY

# Cargar modelo preentrenado con Food-101
model = resnet50(pretrained=True)
model.eval()

#Transformación de la imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
#Clases de comida (Food-101) usando requests
response = requests.get("https://raw.githubusercontent.com/taspinar/tensorflow_food_classification/master/food-101/classes.txt")
if response.status_code == 200:
    food_classes = [line.strip() for line in response.text.splitlines()]
else:
    food_classes = ["unknown"] * 101  # fallback en caso de error
