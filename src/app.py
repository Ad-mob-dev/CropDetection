from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load model
model = load_model("../model/crop_model.h5")

# Class labels
class_labels = {0: 'Bannana', 1: 'Cotton', 2: 'Cucumber',3 : 'Lemon', 4:'Rice' , 5 : "Tomato" , 6: "Wheat"}

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    return jsonify({"prediction": class_labels[class_index]})

if __name__ == '__main__':
    app.run(debug=True)
