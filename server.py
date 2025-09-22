from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
from torchvision import models
from torchvision.models import ResNet50_Weights

app = Flask(__name__)
CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Mapping of internal names to display names ---
display_names = {
    'brain_glioma': 'Brain Glioma',
    'brain_menin': 'Brain Meningioma',
    'brain_tumor': 'Brain Tumor',
}

# --- Main Disease Model Loading ---
class_labels = ['brain_glioma', 'brain_menin', 'brain_tumor']
num_classes = len(class_labels)

model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load("brain_cancer_model.pth", map_location=device))
model.to(device)
model.eval()

# --- Transformation Pipeline ---
main_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        image_bytes = file.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        main_image_tensor = main_transform(image_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(main_image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidences, predicted_indices = torch.topk(probabilities, k=1)

            main_predicted_label_internal = class_labels[predicted_indices[0].item()]
            main_predicted_confidence = confidences[0].item()

        # Get the friendly name from the mapping
        main_predicted_label_display = display_names.get(main_predicted_label_internal, 'Unknown Diagnosis')

        response_data = {
            "prediction": main_predicted_label_display,
            "confidence": f"{main_predicted_confidence:.4f}",
        }

        return jsonify(response_data)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "An internal server error occurred during analysis.", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)