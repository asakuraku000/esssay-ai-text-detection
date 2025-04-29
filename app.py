from flask import Flask, request, jsonify
import re
import os
import uuid
import json
import tensorflow as tf
import numpy as np
import pickle
from datetime import datetime
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Make sure necessary directories exist
os.makedirs('results', exist_ok=True)

# Global variables for model and tokenizer
model = None
tokenizer = None
max_sequence_length = 500

# Text preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters, keeping only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

# Function to split text into chunks
def split_into_chunks(text, chunk_size=350, overlap=50):
    """Split text into overlapping chunks of approximately chunk_size words."""
    words = text.split()
    chunks = []

    if len(words) <= chunk_size:
        return [text]

    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)    
        i += chunk_size - overlap  # Move forward with overlap

    return chunks

# Function to load the model and tokenizer
def load_model_and_tokenizer():
    try:
        global model, tokenizer
        # Load model
        model = tf.keras.models.load_model('roberta_lite_model.keras')

        # Load tokenizer
        with open('roberta_lite_tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        print("Model and tokenizer loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return False

# Function to predict using the loaded model
def predict_with_model(text):
    """
    Make predictions using the loaded TensorFlow model.
    """
    global model, tokenizer, max_sequence_length

    # Preprocess the text
    processed_text = preprocess_text(text)

    # Split into chunks for longer texts
    chunks = split_into_chunks(processed_text)

    chunk_details = []
    chunk_predictions = []

    for i, chunk in enumerate(chunks):
        # Tokenize and pad the sequence
        sequence = tokenizer.texts_to_sequences([chunk])
        padded = pad_sequences(sequence, maxlen=max_sequence_length)

        # Make prediction
        prediction = float(model.predict(padded)[0][0])

        # Calculate probabilities
        ai_probability = prediction
        human_probability = 1 - ai_probability

        chunk_predictions.append(ai_probability)

        # Store details for each chunk
        chunk_details.append({
            "chunk_id": i + 1,
            "text": chunk[:100] + "..." if len(chunk) > 100 else chunk,  # Preview of chunk
            "ai_probability": ai_probability * 100,
            "human_probability": human_probability * 100
        })

    # Calculate average prediction across all chunks
    avg_ai_prob = sum(chunk_predictions) / max(len(chunk_predictions), 1)

    # Determine overall classification
    if avg_ai_prob > 0.5:
        classification = "AI-generated"
        confidence = avg_ai_prob * 100
    else:
        classification = "Human-written"
        confidence = (1 - avg_ai_prob) * 100

    return {
        "classification": classification,
        "confidence": confidence,
        "ai_probability": avg_ai_prob * 100,
        "human_probability": (1 - avg_ai_prob) * 100,
        "chunk_details": chunk_details
    }

# Initialize function to be called during app startup
def initialize_app():
    global model, tokenizer

    if model is None or tokenizer is None:
        print("Initializing AI Detection API with TensorFlow model...")
        success = load_model_and_tokenizer()
        if not success:
            print("WARNING: Failed to load model. Using simulation mode.")

    return True

# API routes
@app.route('/api/analyze', methods=['POST'])
def analyze_essay():
    # Ensure initialization
    initialize_app()

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    if 'essay' not in data:
        return jsonify({"error": "No essay provided"}), 400

    essay = data.get('essay', '')

    if len(essay) < 50:
        return jsonify({"error": "Essay too short for accurate analysis (minimum 50 characters)"}), 400

    # Generate a unique ID for this analysis
    analysis_id = str(uuid.uuid4())

    # Get prediction from model
    if model is not None and tokenizer is not None:
        result = predict_with_model(essay)
    else:
        # Fallback to simulation if model failed to load
        return jsonify({"error": "Model not available. Please check server logs."}), 500

    # Add timestamp and ID
    result['timestamp'] = datetime.now().isoformat()
    result['analysis_id'] = analysis_id

    # Save the result
    with open(f'results/{analysis_id}.json', 'w') as f:
        json.dump(result, f)

    # Return the analysis
    return jsonify(result)

@app.route('/api/results/<analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    try:
        with open(f'results/{analysis_id}.json', 'r') as f:
            result = json.load(f)
        return jsonify(result)
    except FileNotFoundError:
        return jsonify({"error": "Analysis not found"}), 404

@app.route('/api/health', methods=['GET'])
def health_check():
    # Check if model is loaded
    model_status = "loaded" if model is not None else "not loaded"
    tokenizer_status = "loaded" if tokenizer is not None else "not loaded"

    return jsonify({
        "status": "healthy" if model is not None and tokenizer is not None else "degraded",
        "model_status": model_status,
        "tokenizer_status": tokenizer_status
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "service": "AI Text Detection API",
        "status": "running",
        "model_loaded": model is not None and tokenizer is not None,
        "endpoints": {
            "/api/analyze": "POST - Submit an essay for analysis",
            "/api/results/<analysis_id>": "GET - Retrieve a previously analyzed result",
            "/api/health": "GET - Check service health"
        }
    })

# Initialize the app at startup
with app.app_context():
    initialize_app()

if __name__ == '__main__':
    app.run(debug=True)
