import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load pre-trained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers to retain the learned features
for layer in base_model.layers:
    layer.trainable = False

# Add custom top layers for gender classification
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)  # Add a dense layer with 128 units and ReLU activation
x = Dense(64, activation='relu')(x)   # Add another dense layer with 64 units and ReLU activation
output = Dense(1, activation='sigmoid')(x)  # Final layer for binary classification (Male/Female)

# Create the full model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Prepare the dataset (using ImageDataGenerator for data augmentation)
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')

# Assuming you have a 'male' and 'female' folder inside the 'Imagestotrain' directory
train_generator = train_datagen.flow_from_directory(
    'C:/Users/visha/OneDrive/Desktop/Vishal works/Imagestotrain',  # Path to the training dataset
    target_size=(224, 224),  # VGG16 input size
    batch_size=32,
    class_mode='binary'  # Binary classification
)

# Train the model
model.fit(train_generator, epochs=10, steps_per_epoch=100)

# Save the trained model
model.save('gender_classification_model.h5')
