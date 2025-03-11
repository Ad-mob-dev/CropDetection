import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# Paths
train_dir = r"D:\Python\CropDetection\dataset\train"
val_dir = r"D:\Python\CropDetection\dataset\validation"
save_path = r"D:\Python\CropDetection\model\crop_model.h5"

# Get Number of Classes
num_classes = len(os.listdir(train_dir))  # Count folders as class count
print(f"Number of Classes: {num_classes}")

# Load Pretrained ResNet50 Model with Higher Input Resolution
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Unfreeze More Layers for Better Learning
for layer in base_model.layers[-80:]:  # Adjust for deeper feature extraction
    layer.trainable = True

# Add Custom Fully Connected Layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)  # Helps with stable training
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()
x = Dropout(0.3)(x)
output_layer = Dense(num_classes, activation='softmax')(x)

# Create Model
model = Model(inputs=base_model.input, outputs=output_layer)

# **Improved Data Augmentation**
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,  # Stronger rotation
    zoom_range=0.5,     # More zoom variety
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,  # Helps for certain datasets
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(244, 244), batch_size=64, class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(244, 244), batch_size=64, class_mode='categorical'
)

# **Compute Class Weights to Handle Imbalance**
labels = train_generator.classes
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
class_weight_dict = dict(enumerate(class_weights))
print("Computed Class Weights:", class_weight_dict)

# **Advanced Callbacks**
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-7)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# **Use SGD Instead of Adam for Higher Accuracy**
optimizer = SGD(learning_rate=1e-4, momentum=0.9, nesterov=True)

# Compile Model with Categorical Crossentropy Loss
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# **Train Model**
history = model.fit(train_generator, validation_data=val_generator, epochs=50,
                    class_weight=class_weight_dict, callbacks=[reduce_lr, early_stopping])

# **Save Model**
model.save(save_path)
print(f"Model saved successfully at {save_path}")
