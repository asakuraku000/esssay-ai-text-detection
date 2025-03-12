from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import uuid
import json
from datetime import datetime

app = Flask(__name__)

# Make sure necessary directories exist
os.makedirs('results', exist_ok=True)

# Download necessary NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Global variables for model and tokenizer
model = None
tokenizer = None

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

# Enhanced prediction function for long texts
def predict_long_text(text, max_length=400, chunk_size=350, overlap=50):
    global model, tokenizer
    
    # Ensure model and tokenizer are loaded
    if model is None or tokenizer is None:
        load_model_for_prediction()
    
    # Preprocess the text
    processed_text = preprocess_text(text)

    # Split into chunks
    chunks = split_into_chunks(processed_text, chunk_size, overlap)

    # Get predictions for each chunk
    chunk_predictions = []
    chunk_details = []

    for i, chunk in enumerate(chunks):
        # Tokenize and pad
        sequence = tokenizer.texts_to_sequences([chunk])
        padded = pad_sequences(sequence, maxlen=max_length)

        # Predict
        prediction = model.predict(padded, verbose=0)[0][0]
        ai_probability = float(prediction)
        human_probability = 1 - ai_probability

        chunk_predictions.append(ai_probability)

        # Store details for each chunk
        chunk_details.append({
            "chunk_id": i + 1,
            "text": chunk[:100] + "..." if len(chunk) > 100 else chunk,  # Preview of chunk
            "ai_probability": ai_probability * 100,
            "human_probability": human_probability * 100
        })

    # Calculate average and weighted predictions
    avg_ai_prob = sum(chunk_predictions) / len(chunk_predictions)

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

# Function to load the model and tokenizer
def load_model_for_prediction():
    global model, tokenizer
    try:
        # Load model
        model = tf.keras.models.load_model('roberta_lite_model.keras')

        # Load tokenizer
        with open('roberta_lite_tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        return True
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return False

# API routes
@app.route('/api/analyze', methods=['POST'])
def analyze_essay():
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
    
    # Analyze the essay
    result = predict_long_text(essay)
    
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
    global model, tokenizer
    
    if model is None or tokenizer is None:
        status = load_model_for_prediction()
        if status:
            return jsonify({"status": "healthy", "model_loaded": True})
        else:
            return jsonify({"status": "unhealthy", "model_loaded": False}), 500
    
    return jsonify({"status": "healthy", "model_loaded": True})

# Initialize the model when the application starts
@app.before_first_request
def initialize():
    load_model_for_prediction()

if __name__ == '__main__':
    # Load model on startup
    print("Loading AI detection model...")
    model_loaded = load_model_for_prediction()
    
    if model_loaded:
        print("Model loaded successfully!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please ensure model files exist.")
