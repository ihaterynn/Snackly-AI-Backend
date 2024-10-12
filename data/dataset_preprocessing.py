import cv2
import os
import shutil
import random

def preprocess_images(input_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each folder (Nasi Lemak and Roti Canai)
    for food_type in ['nasi_lemak', 'roti_canai']:
        food_path = os.path.join(input_dir, food_type)
        processed_path = os.path.join(output_dir, f"{food_type}_processed")

        os.makedirs(processed_path, exist_ok=True)  # Create folder for processed images

        # Loop through images in the food folder
        for img_name in os.listdir(food_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filter non-image files
                continue

            img_path = os.path.join(food_path, img_name)

            # Read the image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Image {img_path} could not be read.")
                continue

            # Resize the image
            img_resized = cv2.resize(img, (224, 224))

            # Save the processed image
            output_path = os.path.join(processed_path, img_name)
            cv2.imwrite(output_path, img_resized)
            print(f"Processed and saved: {output_path}")

        # Check the number of processed images
        processed_count = len(os.listdir(processed_path))
        print(f"Total processed images for {food_type}: {processed_count}")

def create_dataset_structure(output_base_dir):
    # Create subdirectories for train, validation, and test
    os.makedirs(os.path.join(output_base_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, 'test'), exist_ok=True)

    # Move the processed images into the respective directories for each food type
    for food_type in ['nasi_lemak', 'roti_canai']:
        processed_path = os.path.join(output_base_dir, f"{food_type}_processed")
        images = os.listdir(processed_path)
        random.shuffle(images)  # Shuffle the images for randomness

        # Determine split sizes
        num_images = len(images)
        num_test_images = num_images // 5  # 20% for test
        num_val_images = num_images // 10   # 10% for validation
        num_train_images = num_images - (num_test_images + num_val_images)

        # Create specific directories for the food type
        train_food_dir = os.path.join(output_base_dir, 'train', food_type)
        validation_food_dir = os.path.join(output_base_dir, 'validation', food_type)
        test_food_dir = os.path.join(output_base_dir, 'test', food_type)

        os.makedirs(train_food_dir, exist_ok=True)
        os.makedirs(validation_food_dir, exist_ok=True)
        os.makedirs(test_food_dir, exist_ok=True)

        # Move images to test
        for img_name in images[:num_test_images]:
            img_path = os.path.join(processed_path, img_name)
            shutil.move(img_path, os.path.join(test_food_dir, img_name))

        # Move images to validation
        for img_name in images[num_test_images:num_test_images + num_val_images]:
            img_path = os.path.join(processed_path, img_name)
            shutil.move(img_path, os.path.join(validation_food_dir, img_name))

        # Move images to training
        for img_name in images[num_test_images + num_val_images:]:
            img_path = os.path.join(processed_path, img_name)
            shutil.move(img_path, os.path.join(train_food_dir, img_name))

        # Print counts for each dataset
        print(f"{food_type} - Total images: {num_images}, "
              f"Train: {len(os.listdir(train_food_dir))}, "
              f"Validation: {len(os.listdir(validation_food_dir))}, "
              f"Test: {len(os.listdir(test_food_dir))}")

    # Check for equality in counts
    train_count_nasi = len(os.listdir(os.path.join(output_base_dir, 'train', 'nasi_lemak')))
    train_count_roti = len(os.listdir(os.path.join(output_base_dir, 'train', 'roti_canai')))
    val_count_nasi = len(os.listdir(os.path.join(output_base_dir, 'validation', 'nasi_lemak')))
    val_count_roti = len(os.listdir(os.path.join(output_base_dir, 'validation', 'roti_canai')))
    test_count_nasi = len(os.listdir(os.path.join(output_base_dir, 'test', 'nasi_lemak')))
    test_count_roti = len(os.listdir(os.path.join(output_base_dir, 'test', 'roti_canai')))

    print(f"Final Counts: Nasi Lemak - Train: {train_count_nasi}, Validation: {val_count_nasi}, Test: {test_count_nasi}")
    print(f"Final Counts: Roti Canai - Train: {train_count_roti}, Validation: {val_count_roti}, Test: {test_count_roti}")

if __name__ == "__main__":
    dataset_path = r"C:\Users\User\OneDrive\Desktop\DATASETS\malaysian_food"  # Input directory
    output_path = r"C:\Users\User\OneDrive\Desktop\DATASETS\malaysian_food_processed"  # Output base directory
    preprocess_images(dataset_path, output_path)

    # Create dataset structure with images in train, validation, and test directories
    create_dataset_structure(output_path)
