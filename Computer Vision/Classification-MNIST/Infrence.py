import tensorflow as tf
import cv2

model = tf.keras.models.load_model('best_model_mnist.keras')

# Function for predict Image
def predict(image_path , showImage = False):
    image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    imageresized = cv2.resize(image, (28, 28))

    image = imageresized / 255.0

    image = image.reshape(1, 28, 28, 1)

    prediction = model.predict(image)

    predicted_class = tf.argmax(prediction, axis=1)[0].numpy()
    if showImage:
        cv2.imshow('Show Image' , img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return predicted_class, imageresized
    else:
        return predicted_class

# test model
test_image = 'sample test image/test Image 0.png'

predicted_class= predict(test_image)

print(f"Predicted class: {predicted_class}")