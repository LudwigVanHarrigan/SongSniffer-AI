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
import librosa
import soundfile as sf

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
model.load_state_dict(torch.load("SimpleCNN1_T4.pth", map_location=device, weights_only=True))
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

        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)

        # Subdivide audio into 5-second segments
        segment_length = 5.0  # seconds
        num_segments = int(len(y) / (segment_length * sr))
        predictions = []
        confidences = []

        for i in range(num_segments):
            start_sample = int(i * segment_length * sr)
            end_sample = int((i + 1) * segment_length * sr)
            segment = y[start_sample:end_sample]

            # Save segment as temporary file
            segment_path = os.path.join('uploads', f'segment_{i}.wav')
            sf.write(segment_path, segment, sr)

            # Convert segment to spectrogram
            spectrogram_path = os.path.join('uploads', f'segment_{i}.png')
            generate_mel_spectrogram(segment_path, spectrogram_path, resolution=224)

            # Use the spectrogram for prediction
            image = Image.open(spectrogram_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                predictions.append(predicted_class)
                confidences.append(probabilities[0][predicted_class].item())

        # Perform majority vote
        from collections import Counter
        vote_counts = Counter(predictions)
        majority_vote = vote_counts.most_common(1)[0][0]
        majority_confidence = sum(conf for pred, conf in zip(predictions, confidences) if pred == majority_vote) / vote_counts[majority_vote]

        # Return prediction statistics
        return jsonify({
            'filename': file.filename,
            'prediction': class_names[majority_vote],
            'confidence': majority_confidence,
            'vote_counts': vote_counts
        })
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
    app.run(debug=True, host='0.0.0.0', port=8083)