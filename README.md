# Crop Detection & Image Classification 🌾📷

This project focuses on **crop detection and classification** using deep learning models, specifically leveraging **ResNet50** for improved accuracy. The model is fine-tuned to classify crop images based on a custom dataset.

## Features ✨
- **Pretrained ResNet50** with transfer learning.
- **Advanced data augmentation** for better generalization.
- **Class weight balancing** to handle imbalanced datasets.
- **SGD optimizer with momentum** for better convergence.
- **ReduceLROnPlateau & EarlyStopping** for adaptive learning rate adjustments.
- **Model saving & reusability** for further improvements.

## Installation 🛠️
Ensure you have Python installed, then install the required dependencies:
```sh
pip install tensorflow numpy scikit-learn opencv-python matplotlib
```

## Dataset 📂
The dataset should be structured as:
```
Dataset/
│── train/
│   ├── class_1/
│   ├── class_2/
│   ├── ...
│── validation/
│   ├── class_1/
│   ├── class_2/
│   ├── ...
```
Place the dataset inside `D:\Python\CropDetection\dataset\` for automatic loading.

## Model Training 🚀
Train the model using the following script:
```python
python train_model.py
```
This script:
1. Loads **ResNet50** with pre-trained weights.
2. Unfreezes the last 80 layers for fine-tuning.
3. Applies **strong data augmentation**.
4. Computes **class weights** dynamically.
5. Trains using **SGD optimizer with momentum**.
6. Saves the trained model as `crop_model.h5`.

## Usage 🌱
1. **Prepare Dataset:** Ensure your images are categorized into labeled folders.
2. **Train Model:** Run `train.py` to train and save the model.
3. **Classify Images:** Load the saved model and pass an image for prediction.
4. **Deploy Model:** Use the trained model in an application for real-time classification.

## Model Evaluation 📊
During training, the validation accuracy and loss are monitored. The model is saved at the best performance point using `EarlyStopping`.

## Performance Optimization ⚡
- **Use Transfer Learning** for faster and more efficient training.
- **Data Augmentation** to improve generalization.
- **Optimize inference speed** using TensorRT or ONNX.
- **Adjust hyperparameters** for better results.

## License 📜
This project is licensed under the **MIT License**.

## Contributing 🤝
Feel free to submit issues or pull requests to improve the project!

