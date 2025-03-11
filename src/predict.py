import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("../model/crop_model.h5")

# Class labels
class_labels = {0: 'Bannana', 1: 'Cotton', 2: 'Cucumber',3 : 'Lemon', 4:'Rice' , 5 : "Tomato" , 6: "Wheat"}

def predict_crop(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)


  
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    print(prediction)
    print(f"Prediction: {class_labels[class_index]}")

# Test with an image
predict_crop('../test_image.jpeg')