# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.regularizers import l2  # type: ignore
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# Configure image generators for training and testing
train_generator = ImageDataGenerator(
    rescale=1./255.,
    shear_range=0.1,
    zoom_range=0.3,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

test_generator = ImageDataGenerator(rescale=1./255.)

# Load training and testing data from directories
train_data = train_generator.flow_from_directory(
    'Data/train',
    target_size=(500, 500),
    color_mode='grayscale',
    class_mode='binary',
    batch_size=32
)

test_data = test_generator.flow_from_directory(
    'Data/test',
    target_size=(500, 500),
    color_mode='grayscale',
    class_mode='binary',
    batch_size=32
)

# Check the count of training and testing samples
num_train_samples = len(train_data)
num_test_samples = len(test_data)

print(f'Number of training samples: {num_train_samples}')
print(f'Number of test samples: {num_test_samples}')

# Set up model checkpointing to save the best model based on validation loss
checkpoint = ModelCheckpoint(
    'best_model_xray.keras',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

# Build the Convolutional Neural Network (CNN) model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# Compile the model with Adam optimizer and binary crossentropy loss
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display the model summary
model.summary()

# Compute class weights to handle class imbalance
weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(zip(np.unique(train_data.classes), weights))

# Train the model with training data and validation data
history = model.fit(
    train_data,
    epochs=25,
    validation_data=test_data,
    callbacks=[checkpoint],
    class_weight=class_weights
)

# Plot training history for accuracy
plt.plot(history.history['val_accuracy'],
         color='orange', label='Validation Accuracy')
plt.plot(history.history['accuracy'], color='black', label='Training Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('HistoryOfModel.png')
plt.show()
