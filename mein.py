import gradio as gr
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import openai
import requests

# -------- CONFIGURA TU API KEY Y ORGANIZATION ID DE OPENAI --------
OPENAI_API_KEY = "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"
OPENAI_ORG_ID = "org-Gpagyx25nqJdVGVZUTiZBGUY"

# ✅ CONFIGURACIÓN CORRECTA
openai.api_key = OPENAI_API_KEY
openai.organization = OPENAI_ORG_ID

# Cargar el modelo ResNet50
model = resnet50(pretrained=True)
model.eval()

# Transformación de imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Clases del dataset Food-101
response = requests.get("https://raw.githubusercontent.com/taspinar/tensorflow_food_classification/master/food-101/classes.txt")
if response.status_code == 200:
    food_classes = [line.strip() for line in response.text.splitlines()]
else:
    food_classes = ["unknown"] * 101

# Clasificación de la imagen
def classify_food(img):
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    label = food_classes[predicted.item()]
    return label

# Información nutricional estimada
def simulate_nutrition(food_name):
    base = abs(hash(food_name)) % 300 + 100
    return {
        "Calorías": f"{base} kcal",
        "Proteínas": f"{base // 10} g",
        "Grasas": f"{base // 12} g",
        "Carbohidratos": f"{base // 8} g"
    }

# Solicitar receta a OpenAI
def generate_recipe_with_ai(food_name):
    prompt = f"Dame una receta sencilla y nutritiva usando {food_name} como ingrediente principal."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Cambia a "gpt-4" si tienes acceso
            messages=[
                {"role": "system", "content": "Eres un chef experto en recetas nutritivas."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7,
        )
        return response['choices'][0]['message']['content']
    except openai.error.OpenAIError as e:
        return f"Error al llamar a la API de OpenAI:\n{str(e)}"

# Chat libre con la IA
def chat_with_ai(user_message):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Cambia a "gpt-4" si tienes acceso
            messages=[
                {"role": "system", "content": "Eres un experto en nutrición y comida."},
                {"role": "user", "content": user_message}
            ],
            max_tokens=300,
            temperature=0.7,
        )
        return response['choices'][0]['message']['content']
    except openai.error.OpenAIError as e:
        return f" Error en respuesta IA:\n{str(e)}"

# Función principal
def food_app(image, user_message=""):
    label = classify_food(image)
    nutrition = simulate_nutrition(label)
    recipe = generate_recipe_with_ai(label)

    extra_info = ""
    if user_message.strip():
        extra_info = chat_with_ai(user_message)

    result = f"##  Comida Detectada: **{label}**\n\n"
    result += "###  Información Nutricional Estimada:\n"
    for k, v in nutrition.items():
        result += f"- {k}: {v}\n"
    result += f"\n###  Receta Sugerida:\n{recipe}\n"
    if extra_info:
        result += f"\n###  Respuesta de la IA:\n{extra_info}"
    return result

# Interfaz Gradio
interface = gr.Interface(
    fn=food_app,
    inputs=[
        gr.Image(type="pil", label="📷 Sube una imagen de comida"),
        gr.Textbox(label=" Pregunta o mensaje para la IA (opcional)", placeholder="Ej: ¿Esta comida es saludable?")
    ],
    outputs="markdown",
    title=" NutriFood AI",
    description="Sube una imagen y recibe una receta, información nutricional y una respuesta de la IA. Funciona con tu API Key y Organization ID de OpenAI."
)

interface.launch()
