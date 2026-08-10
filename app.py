import os
import cv2
import sqlite3
import datetime
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import io
from PIL import Image

# ================= PATHS =================
MODEL_PATH = "best_model_35k.pth"  # Your downloaded model
DB_PATH = "predictions.db"     # Database file

# ================= PREPROCESS FUNCTION (from your code) =================
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = enhanced_gray
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img

# ================= TRANSFORMS (from your code, no augmentations for inference) =================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ================= LOAD MODEL =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.densenet201()
num_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 5)  # 0–4 DR levels
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()  # Set to inference mode

# ================= DATABASE SETUP =================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  image_name TEXT,
                  prediction INTEGER,
                  timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()  # Create DB if not exists

def store_prediction(username, image_name, prediction):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO predictions (username, image_name, prediction, timestamp) VALUES (?, ?, ?, ?)",
              (username, image_name, prediction, timestamp))
    conn.commit()
    conn.close()

# ================= STREAMLIT FRONT-END =================
st.title("Diabetic Retinopathy Detection")

# User input
username = st.text_input("Enter your username", value="Anonymous")
uploaded_file = st.file_uploader("Upload a fundus image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the bytes **once** and keep them in memory
    bytes_data = uploaded_file.read()

    # Option A: Display using PIL (recommended, clean)
    image = Image.open(io.BytesIO(bytes_data))
    st.image(image, caption="Uploaded Fundus Image", width=500)  # or use_column_width → width=None for auto

    # Option B: If you prefer OpenCV for display (less common)
    # nparr = np.frombuffer(bytes_data, np.uint8)
    # img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    # st.image(img_cv_rgb, caption="Uploaded Fundus Image", width=500)

    # Now process for model (using the same bytes)
    nparr = np.frombuffer(bytes_data, np.uint8)
    img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_array is None:
        st.error("Failed to decode the image. Please upload a valid JPEG/PNG file.")
    else:
        # Continue with preprocessing
        preprocessed = preprocess_image(img_array)
        transformed = transform(preprocessed).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(transformed)
            _, predicted = torch.max(output, 1)
            dr_level = predicted.item()

        levels = {
            0: "No Diabetic Retinopathy",
            1: "Mild Diabetic Retinopathy",
            2: "Moderate Diabetic Retinopathy",
            3: "Severe Diabetic Retinopathy",
            4: "Proliferative Diabetic Retinopathy"
        }
        result = levels.get(dr_level, "Unknown")

        st.success(f"**Prediction:** {result} (Level {dr_level})")

        # Store in DB
        store_prediction(username, uploaded_file.name, dr_level)
        st.info("Prediction saved to database.")

# View past predictions (optional feature)
if st.button("View Past Predictions"):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    st.dataframe(df)
    conn.close()