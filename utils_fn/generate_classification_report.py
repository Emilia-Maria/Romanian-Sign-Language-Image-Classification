import numpy as np
from sklearn.metrics import classification_report

def generate_classification_report(model, dataset, class_names):
    """
    Generates and prints a classification report.

    Parameters:
    model (tf.keras.Model): The trained model.
    dataset (tf.data.Dataset): The dataset to predict and evaluate.
    class_names (list): List of class names.
    """
    y_pred = np.argmax(model.predict(dataset), axis=-1)
    y_true = np.concatenate([y for x, y in dataset], axis=0)
    y_true = np.argmax(y_true, axis=-1)

    print(classification_report(y_true, y_pred, target_names=class_names))