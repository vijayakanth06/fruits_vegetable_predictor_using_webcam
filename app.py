import os
import cv2
import base64
import numpy as np
from flask import Flask, Response, request, jsonify, render_template
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
from groq import Groq

app = Flask(__name__)
CORS(app)
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load Groq API key

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

# Capture and stream video from webcam
def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Error: Could not open webcam")

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Convert frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Flask route to stream video to the HTML page
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Capture frame and make prediction
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Error: Could not open webcam")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise Exception("Error: Failed to capture image")

    # Save captured frame
    image_path = "captured_image.jpg"
    cv2.imwrite(image_path, frame)
    return image_path

# Function to recognize fruit or vegetable
def recognize_fruit_or_vegetable(image):
    if not labels:
        raise Exception("No labels loaded")

    if embedding_model is None or label_embeddings is None:
        raise Exception("Embedding model not initialized")

    # Convert image to base64
    with open(image, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")

    # Pass the list of labels to the model for improved accuracy
    label_list = ", ".join(labels)

    # Define message for Groq model (NO system message)
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
                    "text": f"Identify the fruit or vegetable in the image. Return only the name or the closest match from this list: {label_list}. If no match is found, return nothing."
                }
            ]
        }
    ]

    try:
        # Make API call to Groq
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=messages,
            temperature=0.3,
            max_tokens=50,
            top_p=1,
            stream=False,
        )
        prediction = completion.choices[0].message.content.strip()

        # Check if prediction is in labels
        if prediction in labels:
            return prediction

        # If not in labels, find closest match using cosine similarity
        pred_embedding = embedding_model.encode(prediction, convert_to_tensor=True)
        similarity_scores = util.pytorch_cos_sim(pred_embedding, label_embeddings)[0]
        best_match_idx = similarity_scores.argmax().item()
        best_match_score = similarity_scores[best_match_idx].item()

        # Only return match if similarity is above threshold (e.g., 0.7)
        if best_match_score > 0.7:
            return labels[best_match_idx]
        else:
            return None

    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")



# Flask route to predict from captured frame
@app.route("/predict", methods=["POST"])
def predict():
    try:
        image_path = capture_image()
        result = recognize_fruit_or_vegetable(image_path)
        if result:
            return jsonify({"prediction": result})
        else:
            return jsonify({"prediction": ""})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Flask route to serve HTML page
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

