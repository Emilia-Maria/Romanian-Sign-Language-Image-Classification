import numpy as np
from sklearn.metrics import classification_report

def generate_classification_report(model, dataset, features=None, labels=None):
    """
    Generates and prints a classification report.

    Parameters:
    model (tf.keras.Model): The trained model.
    dataset (tf.data.Dataset): The dataset to predict and evaluate.
    features (np.ndarray): The features to predict. Default is None.
    labels (np.ndarray): The true labels. Default is None.
    """
    if features is not None and labels is not None:
      y_pred = np.argmax(model.predict(features), axis=-1)
      if len(labels.shape) > 1 and labels.shape[1] > 1:
        y_true = np.argmax(labels, axis=-1)
      else:
        y_true = labels
    elif dataset is not None:
      y_pred = np.argmax(model.predict(dataset), axis=-1)
      y_true = np.concatenate([y for x, y in dataset], axis=0)
      y_true = np.argmax(y_true, axis=-1)
    else:
      raise ValueError("Either dataset or both features and labels must be provided.")

    target_names = dataset.class_names
    print(classification_report(y_true, y_pred, target_names=target_names))