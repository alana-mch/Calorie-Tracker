import json
import torch
import torch.nn as nn
import gradio as gr
from torchvision import transforms
from torchvision.models import efficientnet_b0

device = "cpu" 

#load files
classes = json.load(open("classes.json"))
calories = json.load(open("calories.json"))

#load model
model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(1280, len(classes))
model.load_state_dict(torch.load("food_model.pth", map_location=device))
model.eval()

#transforms for images for the models
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

THRESHOLD = 0.4   
FLOOR = 0.2      

def predict(img):
    if img is None:
        return {}, "Upload or take a photo of your meal."
    x = tf(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    confidences = {classes[i]: float(probs[i]) for i in range(len(classes))}

    top = probs.topk(4)
    names = [classes[i] for i in top.indices.tolist()]
    p = top.values[0].item()
    food = names[0]
    alternatives = [n for n in names[1:] if n != "not_food"][:2]

    if food == "not_food":
        message = "This doesn't look like food I recognise."
    elif p < FLOOR:
        message = "Not sure what this is."
    elif p < THRESHOLD:
        message = (f"Best guess: {food} (about {calories[food]} kcal), "
                   f"but it could also be {' or '.join(alternatives)}.")
    else:
        message = f"{food}: about {calories[food]} kcal per typical serving"
    return confidences, message

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", sources=["upload", "webcam"], label="Photo of your meal"),
    outputs=[gr.Label(num_top_classes=3, label="Prediction"),
             gr.Textbox(label="Calorie estimate")],
    title="Meal Scanner",
    description=("Recognises 101 foods from the Food-101 dataset and gives an approximate "
                 "calorie estimate for a typical serving. One food per photo; portion size "
                 "is not estimated."),
)

demo.launch()