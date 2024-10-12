from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from models.mobilenet import load_mobilenetv2
from models.mobilevit import MobileViT
from models.mask_rcnn import load_mask_rcnn
import os
import base64

# Initialize Flask app
app = Flask(__name__)

# Load models
print("Loading models...")
model = tf.keras.models.load_model('saved_models/hybrid_model_final')
mobilenet_model = load_mobilenetv2()
mask_rcnn_model = load_mask_rcnn()
print("Models loaded and ready for inference.")

# Nutrition Estimation for different food classes
nutrition_data = {
    "nasi_lemak": {"calories": 644, "carbs": 86, "protein": 17, "fat": 26},
    "roti_canai": {"calories": 300, "carbs": 42, "protein": 8, "fat": 12},
}

# Preprocess image for inference
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize image
    return img_array

# Perform object detection using Mask R-CNN
def perform_object_detection(img_path):
    img = image.load_img(img_path, target_size=(320, 320))  # Load and resize the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image

    detection_fn = mask_rcnn_model.signatures['default']
    result = detection_fn(tf.convert_to_tensor(img_array))

    boxes = result['detection_boxes'].numpy()  # Bounding boxes
    class_ids = result['detection_classes'].numpy() if 'detection_classes' in result else None  # Class IDs
    masks = result['detection_masks'].numpy() if 'detection_masks' in result else None  # Detection masks

    return masks, boxes, class_ids

# Estimate portion size based on the mask or bounding box area
def estimate_portion_size(mask_or_box_area, food_type, image_area):
    standard_portion = {
        "nasi_lemak": {"area": 0.15, "calories": 600},
        "roti_canai": {"area": 0.1, "calories": 300}
    }

    if food_type in standard_portion:
        standard_area = standard_portion[food_type]["area"]
        normalized_area = mask_or_box_area / image_area
        portion_size_ratio = normalized_area / standard_area
        portion_size_ratio = min(max(portion_size_ratio, 0.1), 2.0)
        return portion_size_ratio
    else:
        return None

# Make predictions using the hybrid model (MobileNetV2 + MobileViT + Mask R-CNN)
def make_prediction(img_path, threshold=0.5):
    processed_image = preprocess_image(img_path)

    # Step 1: Get feature maps from MobileNetV2
    mobilenet_features = mobilenet_model(processed_image, training=False)

    # Step 2: Pass features through the Vision Transformer (MobileViT)
    mobilevit_model = MobileViT(num_classes=128)
    vit_output = mobilevit_model(mobilenet_features)

    # Step 3: Perform object detection and handle bounding boxes and masks for portion estimation
    masks, boxes, class_ids = perform_object_detection(img_path)
    image_area = 320 * 320  # Assuming the image is resized to 320x320 for object detection

    if masks is not None:
        mask_area = np.count_nonzero(masks[0])  # Number of non-zero pixels
        portion_size = estimate_portion_size(mask_area, "nasi_lemak", image_area)  # Placeholder for food_type
    else:
        box_area = np.mean([(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]) * image_area
        portion_size = estimate_portion_size(box_area, "nasi_lemak", image_area)  # Placeholder for food_type

    # Step 4: Predict food type using hybrid model (Binary classification logic)
    prediction = model.predict(processed_image)
    predicted_label_idx = 1 if prediction >= threshold else 0
    food_labels = ['nasi_lemak', 'roti_canai']  # Ensure correct order
    predicted_label = food_labels[predicted_label_idx]

    # Step 5: Retrieve nutrition information based on prediction
    nutrition_info = nutrition_data.get(predicted_label, "Unknown food")
    if portion_size:
        adjusted_nutrition = {
            'calories': nutrition_info['calories'] * portion_size,
            'carbs': nutrition_info['carbs'] * portion_size,
            'protein': nutrition_info['protein'] * portion_size,
            'fat': nutrition_info['fat'] * portion_size
        }
    else:
        adjusted_nutrition = {"error": "Could not estimate portion size"}

    return predicted_label, adjusted_nutrition

# Define the homepage
@app.route('/')
def index():
    return '''
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Food Prediction and Nutrition Estimation</title>
    </head>
    <body>
        <h1>Upload an image to get food prediction and nutrition estimation</h1>
        <form action="/infer" method="POST" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <input type="submit" value="Upload Image">
        </form>
    </body>
    </html>
    '''

# Define the inference route
@app.route('/infer', methods=['POST'])
def infer():
    try:
        # Check if base64 image data is provided in the request
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400

        # Decode base64 image
        image_data = data['image']
        img_bytes = base64.b64decode(image_data)

        # Save the decoded image temporarily
        img_path = 'temp_image.jpg'
        with open(img_path, 'wb') as img_file:
            img_file.write(img_bytes)

        # Make the prediction
        label, nutrition = make_prediction(img_path)

        # Clean up the temporary image file
        if os.path.exists(img_path):
            os.remove(img_path)

        return jsonify({
            'Predicted Food': label,
            'Estimated Nutrition Info': nutrition
        })

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": "Failed to process image"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
