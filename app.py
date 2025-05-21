import os
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
from groq import Groq
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Load Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Load labels from labels.txt
try:
    with open("labels.txt", "r", encoding="utf-8") as file:
        labels = [line.strip() for line in file.readlines()]
except Exception as e:
    print(f"Error loading labels: {e}")
    labels = []

# Load embedding model for similarity matching
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    label_embeddings = embedding_model.encode(labels, convert_to_tensor=True)
except Exception as e:
    print(f"Error loading embedding model: {e}")
    embedding_model = None
    label_embeddings = None

# Function to recognize fruit or vegetable
def recognize_fruit_or_vegetable(image):
    if not labels:
        raise Exception("No labels loaded")

    if embedding_model is None or label_embeddings is None:
        raise Exception("Embedding model not initialized")

    # Convert image to base64
    _, buffer = cv2.imencode('.jpg', image)
    base64_image = base64.b64encode(buffer).decode("utf-8")

    label_list = ", ".join(labels)

    # Groq model prompt
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
                    "text": f"Identify the fruit or vegetable in the image. Return only the name or the closest match from this list: {label_list}.Do not include any other text or explanation. Just give the name of the fruit or vegetabe name only as output"
                }
            ]
        }
    ]

    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=0.4,
            max_tokens=100,
            top_p=1,
            stream=False,
        )
        prediction = completion.choices[0].message.content.strip()

        # Check if prediction is in labels
        if prediction in labels:
            return prediction

        # If not in labels, find closest match
        pred_embedding = embedding_model.encode(prediction, convert_to_tensor=True)
        similarity_scores = util.pytorch_cos_sim(pred_embedding, label_embeddings)[0]
        best_match_idx = similarity_scores.argmax().item()
        best_match_score = similarity_scores[best_match_idx].item()
        
        if best_match_score > 0.5:
            return labels[best_match_idx]
        else:
            return None
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")

# Flask route to predict from uploaded frame
@app.route("/predict", methods=["POST"])
def predictmodel():
    try:
        file = request.files["image"]
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise Exception("Invalid image format")

        result = recognize_fruit_or_vegetable(image)
        if result:
            return jsonify({"prediction": result})
        else:
            return jsonify({"prediction": ""})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Home route
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port,debug=True)