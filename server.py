# මේ project එක web app එකක් විදිහට පාවිච්චි කරන්න අවශ්‍ය libraries ටික import කරගමු.
# This imports the necessary libraries to use the project as a web app.
from flask import Flask, request, jsonify, render_template # Flask library එකෙන් තමයි web server එක හදන්නේ.
from flask_cors import CORS # Front-end එකෙන් API එකට request යවද්දී security issues වළක්වන්න මේක උදව් වෙනවා.
import torch # Deep learning model එක load කරන්න සහ predict කරන්න ඕන වෙනවා.
import torch.nn as nn
import torchvision.transforms as transforms # Images predict කරන්න කලින් සූදානම් කරන්න.
from PIL import Image # Images handle කරන්න පුළුවන් library එකක්.
import io # Image එක file එකක් විදිහට read කරන්න.
import numpy as np
from torchvision import models
from torchvision.models import ResNet50_Weights

# Flask web server එක හදමු.
app = Flask(__name__)
# CORS enable කරමු, එතකොට ඕනෑම තැනක ඉඳන් API එකට access කරන්න පුළුවන්.
CORS(app)

# GPU එකක් (cuda) තියෙනවද කියලා බලලා, නැත්නම් CPU එක පාවිච්චි කරන්න කියලා define කරගමු.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- අභ්‍යන්තර නම් ප්‍රදර්ශනය කරන නම් වලට සිතියම්ගත කිරීම ---
# --- Mapping of internal names to display names ---
# Project එකේ results usersට ලේසියෙන් තේරුම් ගන්න පුළුවන් විදියට display කරන්න මේක උදව් වෙනවා.
display_names = {
    'brain_glioma': 'Brain Glioma',
    'brain_menin': 'Brain Meningioma',
    'brain_tumor': 'Brain Tumor',
}

# --- ප්‍රධාන රෝග ආකෘතිය පැටවීම ---
# --- Main Disease Model Loading ---
# අපි train කරපු model එක load කරගමු.
class_labels = ['brain_glioma', 'brain_menin', 'brain_tumor'] # Classes වල නම් define කරමු.
num_classes = len(class_labels)

# ResNet50 model එක load කරලා අපේ classes ගානට අනුව layer එක හදමු.
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
# Train කරපු model එකේ weights load කරමු.
model.load_state_dict(torch.load("brain_cancer_model.pth", map_location=device))
# Model එක GPU එකට හෝ CPU එකට යවමු.
model.to(device)
# Model එක evaluation mode එකට දාමු.
model.eval()

# --- Transformation Pipeline ---
# Images predict කරන්න කලින් අවශ්‍ය විදියට වෙනස් කරගමු.
main_transform = transforms.Compose([
    transforms.Resize((224, 224)), # Images ඔක්කොම 224x224 size එකට හරවමු.
    transforms.ToTensor(), # Image එක PyTorch tensor එකක් බවට convert කරගමු.
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Image එක normalize කරමු.
])

# '/' (home) route එක define කරමු.
@app.route('/')
def home():
    # 'index.html' file එක load කරමු.
    return render_template('index.html')

# '/predict' (prediction) route එක define කරමු. මේක POST request එකකින් වැඩ කරනවා.
@app.route('/predict', methods=['POST'])
def predict():
    # File එකක් upload කළාද කියලා බලමු.
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        # File එක bytes විදිහට කියවමු.
        image_bytes = file.read()
        # bytes වලින් PIL (Pillow) image එකක් හදමු.
        image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Image එක transform කරලා model එකට දෙන්න සූදානම් කරගමු.
        main_image_tensor = main_transform(image_pil).unsqueeze(0).to(device)

        # Gradient ගණනය කරන්නේ නැති බව කියනවා (memory save කරගන්න).
        with torch.no_grad():
            # Model එකෙන් prediction එක අරගමු.
            outputs = model(main_image_tensor)
            # Outputs probabilities බවට convert කරමු.
            probabilities = torch.softmax(outputs, dim=1)[0]
            # වැඩිම probability එක තියෙන එක තෝරගමු.
            confidences, predicted_indices = torch.topk(probabilities, k=1)

            # Predictions වලට අදාල නම් හොයාගමු.
            main_predicted_label_internal = class_labels[predicted_indices[0].item()]
            main_predicted_confidence = confidences[0].item()

        # friendly name එකක් හොයාගමු.
        main_predicted_label_display = display_names.get(main_predicted_label_internal, 'Unknown Diagnosis')

        # Result එක JSON format එකට හදමු.
        response_data = {
            "prediction": main_predicted_label_display,
            "confidence": f"{main_predicted_confidence:.4f}",
        }

        # Result එක client එකට යවමු.
        return jsonify(response_data)

    except Exception as e:
        # Error එකක් ආවොත්, ඒක print කරලා error message එකක් client එකට යවමු.
        print(f"Error during prediction: {e}")
        return jsonify({"error": "An internal server error occurred during analysis.", "details": str(e)}), 500

# මේ file එක direct run කළාම server එක start කරනවා.
if __name__ == '__main__':
    app.run(debug=True, port=5000)
