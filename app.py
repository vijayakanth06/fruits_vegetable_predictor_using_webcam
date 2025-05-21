import os
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
from groq import Groq
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# Initialize with error handling
try:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    client = None

# Load labels with retry logic
def load_labels(max_retries=3):
    labels = []
    for attempt in range(max_retries):
        try:
            with open("labels.txt", "r", encoding="utf-8") as file:
                labels = [line.strip() for line in file.readlines()]
                if labels:
                    return labels
        except Exception as e:
            print(f"Attempt {attempt + 1} failed to load labels: {e}")
            time.sleep(1)
    return labels

labels = load_labels()

# Load embedding model with error handling
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    if labels:
        label_embeddings = embedding_model.encode(labels, convert_to_tensor=True)
    else:
        label_embeddings = None
except Exception as e:
    print(f"Error loading embedding model: {e}")
    embedding_model = None
    label_embeddings = None

def recognize_fruit_or_vegetable(image):
    if not labels:
        raise Exception("No labels loaded")
    if embedding_model is None or label_embeddings is None:
        raise Exception("Embedding model not initialized")

    try:
        _, buffer = cv2.imencode('.jpg', image)
        base64_image = base64.b64encode(buffer).decode("utf-8")
        label_list = ", ".join(labels)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    },
                    {
                        "type": "text",
                        "text": f"Identify the fruit or vegetable in the image. Return only the name or the closest match from this list: {label_list}. Do not include any other text or explanation. Just give the name of the fruit or vegetable as output."
                    }
                ]
            }
        ]

        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=0.4,
            max_tokens=100,
            top_p=1,
            stream=False,
        )
        prediction = completion.choices[0].message.content.strip()

        if prediction in labels:
            return prediction

        pred_embedding = embedding_model.encode(prediction, convert_to_tensor=True)
        similarity_scores = util.pytorch_cos_sim(pred_embedding, label_embeddings)[0]
        best_match_idx = similarity_scores.argmax().item()
        best_match_score = similarity_scores[best_match_idx].item()
        
        return labels[best_match_idx] if best_match_score > 0.5 else None

    except Exception as e:
        raise Exception(f"Prediction failed: {e}")

@app.route("/predict", methods=["POST"])
def predictmodel():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Read image file
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Invalid image format"}), 400

        result = recognize_fruit_or_vegetable(image)
        return jsonify({
            "prediction": result if result else "",
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e),
            "status": "error"
        }), 500

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "groq_ready": client is not None,
        "model_ready": embedding_model is not None,
        "labels_loaded": bool(labels)
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)