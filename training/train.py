import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from models.mobilenet import load_mobilenetv2
from models.mobilevit import MobileViT
from models.mask_rcnn import load_mask_rcnn
from training.data_loader import load_data
import os
import numpy as np
from sklearn.metrics import confusion_matrix

# Load the dataset
batch_size = 8
train_data, val_data, test_data = load_data(batch_size)

# Load the pretrained MobileNetV2 model (CNN as feature extractor)
base_model = load_mobilenetv2()

# Vision Transformer Block (ViT)
vit_block = MobileViT(num_classes=128)  # Modify to accept MobileNetV2 output

# Load Mask R-CNN for portion estimation (you can use it in preprocessing if needed)
mask_rcnn_model = load_mask_rcnn()

# Build the final model combining MobileNetV2 and ViT
inputs = tf.keras.Input(shape=(224, 224, 3))

# Step 1: Pass inputs through MobileNetV2 (CNN feature extraction)
features = base_model(inputs, training=False)  # Feature maps output (4D)

# Step 2: Pass the CNN features through the Vision Transformer (ViT)
vit_output = vit_block(features)  # Use the output directly without extra pooling

# Step 3: Classification head
x = tf.keras.layers.Dense(128, activation='relu')(vit_output)  # Directly use the transformer output
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # Binary classification

# Step 4: Compile the model
model = tf.keras.Model(inputs, output)

# Compile the model with RMSprop optimizer
optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-3)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Callbacks for saving the best model and learning rate adjustment
checkpoint = ModelCheckpoint(filepath='saved_models/hybrid_model', 
                             monitor='val_accuracy', save_best_only=True, verbose=1)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
tensorboard = TensorBoard(log_dir=os.path.join("logs", "fit"))

# Custom callback for confusion matrix after each epoch
class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        true_labels = []
        predicted_labels = []
        for images, labels in val_data:
            preds = self.model.predict(images)
            predicted_labels.extend(np.round(preds).astype(int))
            true_labels.extend(labels.numpy())

        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        print(f"\nConfusion Matrix after Epoch {epoch+1}:")
        print(cm)

# Train the model
history = model.fit(train_data, 
                    validation_data=val_data, 
                    epochs=10,                    ##EPOCH 
                    callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard, ConfusionMatrixCallback()])

# Save the final model
model.save('saved_models/hybrid_model_final', save_format='tf')

# Evaluate the model on the test dataset
test_results = model.evaluate(test_data)

# Print the evaluation results, unpacking all metrics (loss, accuracy, precision, recall)
print(f"Test Results - Loss: {test_results[0]}, Accuracy: {test_results[1]}, Precision: {test_results[2]}, Recall: {test_results[3]}")
