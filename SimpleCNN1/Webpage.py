import flask
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
import os
from torchvision import transforms
from SimpleCNN1 import SimpleCNN
from convert_wav_to_spectrogram import generate_mel_spectrogram
import atexit
import shutil

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
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=2, img_height=224, img_width=224)
model.load_state_dict(torch.load("SimpleCNN1_T1.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

# Function to clear the uploads folder
def clear_uploads_folder():
    uploads_folder = 'uploads'
    if os.path.exists(uploads_folder):
        shutil.rmtree(uploads_folder)
        os.makedirs(uploads_folder)

# Register the function to run at exit
atexit.register(clear_uploads_folder)

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

    # Check file type
    if file.filename.endswith('.wav'):
        # Save and preprocess the audio file
        audio_path = os.path.join('uploads', file.filename)
        file.save(audio_path)

        # Convert to spectrogram
        spectrogram_path = os.path.join('uploads', file.filename.rsplit('.', 1)[0] + '.png')
        generate_mel_spectrogram(audio_path, spectrogram_path, resolution=224)

        # Use the spectrogram for prediction
        image = Image.open(spectrogram_path).convert('RGB')
    else:
        # Assume it's an image file
        img_path = os.path.join('uploads', file.filename)
        file.save(img_path)
        image = Image.open(img_path).convert('RGB')

    # Preprocess the image
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
    app.run(debug=True, host='0.0.0.0', port=8082)