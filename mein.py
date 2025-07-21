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