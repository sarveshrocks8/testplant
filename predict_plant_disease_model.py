# %%
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# ✅ Load saved model
model = load_model("plant_disease_cnn_model.h5")

# ✅ Classes (same order as during training)
# Tumhe ye mapping training ke time se hi lena hoga:
# Example ke liye, agar tumne ye dictionary save nahi ki thi:
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


# ✅ Load and preprocess image
img_path = r"C:\Users\Dell\OneDrive\Pictures\Screenshots\potato early.jpg"
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# ✅ Predict
pred = model.predict(img_array)
class_index = np.argmax(pred, axis=1)[0]
class_label = class_labels.get(class_index, "Unknown")

print("🪴 Predicted Class:", class_label)


# %%
