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

#Función para clasificar imagen
def classify_image(img):
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    label = food_classes[predicted.item()]
    return label
 #Obtener información nutricional y receta desde Spoonacular
def get_nutrition_and_recipe(food_name):
    nutrition_url = f"https://api.spoonacular.com/recipes/guessNutrition?title={food_name}&apiKey={SPOONACULAR_API_KEY}"
    recipe_url = f"https://api.spoonacular.com/recipes/complexSearch?query={food_name}&number=1&addRecipeInformation=true&apiKey={SPOONACULAR_API_KEY}"

    nutri_res = requests.get(nutrition_url).json()
    recipe_res = requests.get(recipe_url).json()

    try:
        nutrients = {
            "Calorías": nutri_res['calories']['value'],
            "Proteínas": nutri_res['protein']['value'],
            "Grasas": nutri_res['fat']['value'],
            "Carbohidratos": nutri_res['carbs']['value'],
        }
    except:
        nutrients = {"Calorías": "N/A", "Proteínas": "N/A", "Grasas": "N/A", "Carbohidratos": "N/A"}

    try:
        recipe = recipe_res['results'][0]['summary']
        recipe = recipe.replace('<b>', '').replace('</b>', '')
    except:
        recipe = "No se pudo generar una receta."

    return nutrients, recipe
 # Chat inteligente con OpenAI (GPT)
def chat_with_ai(message):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": message}],
            max_tokens=200,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error en el chat: {e}"
    
    #  Función principal de la app
def food_app(image, user_message):
    label = classify_image(image)
    nutrients, recipe = get_nutrition_and_recipe(label)
    ai_response = chat_with_ai(user_message if user_message else f"Dame más información sobre {label}")

    result = f" **Comida detectada:** {label}\n\n"
    result += " **Valor nutricional estimado:**\n"
    for k, v in nutrients.items():
        result += f"- {k}: {v}\n"
    result += f"\n **Receta sugerida:**\n{recipe}\n"
    result += f"\n **Chat IA:**\n{ai_response}"
    return result