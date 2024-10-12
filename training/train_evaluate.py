import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from training.data_loader import load_data

# Load the trained model
model = tf.keras.models.load_model('saved_models/hybrid_model')

# Load validation dataset
_, val_data = load_data(batch_size=32)

# Evaluate model
loss, accuracy = model.evaluate(val_data)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Get true labels and predictions
y_true = []
y_pred = []

for images, labels in val_data:
    predictions = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

# Convert lists to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display confusion matrix with title
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=val_data.class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Validation Data")  # Add title
plt.show()
