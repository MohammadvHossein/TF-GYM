import tensorflow as tf
from tensorflow.keras.preprocessing import image  # type: ignore
import numpy as np
import cv2

model = tf.keras.models.load_model('best_model_xray.keras')

# Function for predict Image


def predict(img_path, showImage=False):
    img = image.load_img(img_path, target_size=(
        500, 500), color_mode='grayscale')

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array /= 255.0

    prediction = model.predict(img_array)

    if showImage:
        img = cv2.imread(img_path)
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if prediction[0][0] >= 0.5:
        return "Bacteria"
    else:
        return "Normal"


# test model
test_image = 'sample test image\\BACTERIA 5.jpeg'
# test_image = 'sample test image\\NORMAL 5.jpeg'

predicted_class = predict(test_image , showImage=True)

print(f"Predicted class: {predicted_class}")
