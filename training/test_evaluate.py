import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from training.data_loader import load_data

# Load the trained model
model = tf.keras.models.load_model('saved_models/hybrid_model')

# Load test dataset
test_data = load_data(batch_size=32, is_test=True)

# Check the number of batches and total images in the test dataset
num_batches = len(test_data)
num_images = num_batches * 32  # Assuming batch size of 32
print(f"Total number of batches in test dataset: {num_batches}")
print(f"Estimated total images in test dataset: {num_images}")

# Evaluate model on test data
loss, accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Get true labels and predictions
y_true = np.concatenate([y.numpy() for _, y in test_data])  # Get true labels from the dataset
y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Display confusion matrix with title
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_data.class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Test Data")  # Add title
plt.show()
