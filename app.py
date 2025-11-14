from dotenv import load_dotenv
import google.generativeai as genai
import markdown
from markupsafe import Markup
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# 🌱 Load environment variables
load_dotenv()

# 🌱 Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)

# ✅ AI-generated solution
def get_disease_solution(disease_name):
    prompt = f"""
    You are an agricultural expert. Provide short, practical advice for:
    - Disease: {disease_name}
    - Include bullet points for: cause, symptoms, prevention, and treatment.
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        if not response or not hasattr(response, "text") or not response.text.strip():
            return "⚠️ AI couldn't generate a response. Please try again."

        return response.text.strip()

    except Exception as e:
        print("⚠️ Gemini error:", e)
        return f"⚠️ AI service error: {str(e)}"


# ✅ Load trained CNN model
model = load_model("plant_disease_cnn_model.h5")

# ✅ Class labels
class_labels = {
    0: 'Pepper__bell___Bacterial_spot',
    1: 'Pepper__bell___healthy',
    2: 'Potato___Early_blight',
    3: 'Potato___healthy',
    4: 'Potato___Late_blight',
    5: 'Tomato___Target_Spot',
    6: 'Tomato___Tomato_mosaic_virus',
    7: 'Tomato___Tomato_YellowLeaf_Curl_Virus',
    8: 'Tomato___Bacterial_spot',
    9: 'Tomato___Early_blight',
    10: 'Tomato___healthy',
    11: 'Tomato___Late_blight',
    12: 'Tomato___Leaf_Mold',
    13: 'Tomato___Septoria_leaf_spot',
    14: 'Tomato___Spider_mites_Two_spotted_spider_mite'
}


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="⚠️ No file uploaded!")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction="⚠️ Please select an image file.")

    # Save uploaded file
    file_path = os.path.join('static', file.filename)
    file.save(file_path)

    try:
        # Preprocess image
        img = image.load_img(file_path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        pred = model.predict(img_array)
        class_index = np.argmax(pred, axis=1)[0]
        predicted_class = class_labels[class_index]

        # Get AI solution
        solution_text = get_disease_solution(predicted_class)
        print("✅ AI Response:", solution_text[:200])  # Debug log

        # Markdown to safe HTML
        solution_html = Markup(markdown.markdown(solution_text))

        return render_template(
            'index.html',
            prediction=predicted_class,
            solution=solution_html,
            img_path=file_path
        )

    except Exception as e:
        print("🔥 Prediction error:", e)
        return render_template('index.html', prediction="❌ Error", solution=f"{e}")


if __name__ == '__main__':
    app.run(debug=True)
