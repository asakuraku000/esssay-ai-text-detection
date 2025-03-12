from flask import Flask, request, jsonify
import re
import os
import uuid
import json
import random
from datetime import datetime

app = Flask(__name__)

# Make sure necessary directories exist
os.makedirs('results', exist_ok=True)

# Global variable to track if initialization has happened
initialized = False

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

# Simulated prediction function (since we don't have the actual model)
def simulate_prediction(text):
    """
    This function simulates AI text detection without needing the actual model.
    It uses basic text features to make a rough estimation.
    
    NOTE: This is NOT an accurate detector - it's just a placeholder until
    you can implement the real model.
    """
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Split into chunks
    chunks = split_into_chunks(processed_text)
    
    chunk_details = []
    chunk_predictions = []
    
    for i, chunk in enumerate(chunks):
        # Simple simulation based on text length, word variety, etc.
        words = chunk.split()
        unique_words = set(words)
        
        # Calculate some basic text features
        word_variety = len(unique_words) / max(len(words), 1)  # Unique word ratio
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
        
        # Simulate an AI probability based on these features
        # NOTE: This is NOT scientific, just a placeholder
        base_ai_prob = 0.4 + (0.2 * (1 - word_variety)) + (0.05 * min(avg_word_length/10, 0.1))
        
        # Add some randomness
        noise = random.uniform(-0.15, 0.15)
        ai_probability = max(0.1, min(0.9, base_ai_prob + noise))
        
        human_probability = 1 - ai_probability
        
        chunk_predictions.append(ai_probability)
        
        # Store details for each chunk
        chunk_details.append({
            "chunk_id": i + 1,
            "text": chunk[:100] + "..." if len(chunk) > 100 else chunk,  # Preview of chunk
            "ai_probability": ai_probability * 100,
            "human_probability": human_probability * 100
        })
    
    # Calculate average prediction
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
        "chunk_details": chunk_details,
        "note": "This is a simulated result as the real model is not available."
    }

# Initialize function to be called from routes
def initialize_app():
    global initialized
    if not initialized:
        print("Initializing AI Detection API in SIMULATION mode (no model required)")
        initialized = True
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
    
    # Get simulated prediction
    result = simulate_prediction(essay)
    
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
    # Ensure initialization
    initialize_app()
    
    try:
        with open(f'results/{analysis_id}.json', 'r') as f:
            result = json.load(f)
        return jsonify(result)
    except FileNotFoundError:
        return jsonify({"error": "Analysis not found"}), 404

@app.route('/api/health', methods=['GET'])
def health_check():
    # Ensure initialization
    initialize_app()
    
    return jsonify({
        "status": "healthy", 
        "model": "simulated",
        "note": "Using simulated predictions as real model is not available"
    })

@app.route('/', methods=['GET'])
def root():
    # Ensure initialization
    initialize_app()
    
    return jsonify({
        "service": "AI Text Detection API (Simulation Mode)",
        "status": "running",
        "endpoints": {
            "/api/analyze": "POST - Submit an essay for analysis (simulated)",
            "/api/results/<analysis_id>": "GET - Retrieve a previously analyzed result",
            "/api/health": "GET - Check service health"
        },
        "note": "This version uses simulated predictions as the real model is not available"
    })

# This replaces the removed @app.before_first_request decorator
with app.app_context():
    initialize_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
