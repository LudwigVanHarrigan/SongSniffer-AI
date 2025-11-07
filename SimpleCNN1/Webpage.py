import flask
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
import os
from torchvision import transforms
from SimpleCNN1 import SimpleCNN

# Serve static files
app = Flask(__name__, static_folder='static', static_url_path='/static')

# Define preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Class names
class_names = ['smelly', 'not_smelly']

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=2, img_height=224, img_width=224)
model.load_state_dict(torch.load("SimpleCNN1_T1.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save and preprocess the image
    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)
    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    # Return prediction
    return jsonify({
        'filename': file.filename,
        'prediction': class_names[predicted_class],
        'confidence': confidence
    })

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, port=5001)