import numpy as np
import tensorflow as tf

def predict_image(image_path, model_path, input_size, class_labels):
    """
    Predicts the sign language class from an image using a pre-trained model and prints the confidence.

    Parameters:
    - image_path (str): Path to the image file.
    - model_path (str): Path to the pre-trained model file.
    - input_size (tuple): Size to which the image should be resized.
    - class_labels (list): List of class labels corresponding to model output classes.
    """
    
    model = tf.keras.models.load_model(model_path)

    image = tf.keras.preprocessing.image.load_img(image_path, target_size=input_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0) 

    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index] * 100 

    if class_labels is not None:
        predicted_class = class_labels[class_index]
    else:
        predicted_class = str(class_index)

    print(f'Predicted class - {predicted_class} with a confidence of - {confidence:.3f}%')