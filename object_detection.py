# object_detection.py

# Install required libraries
# !pip install opencv-python
# !pip install torch torchvision  # For PyTorch-based YOLO
# !pip install sqlalchemy         # For database operations

import os
import cv2
import torch
import logging
from typing import List, Dict, Optional
from sqlalchemy import create_engine, Table, Column, Integer, String, Float, MetaData
from sqlalchemy.exc import SQLAlchemyError
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('object_detection.log'),
        logging.StreamHandler()
    ]
)

# Configuration (could be moved to a config file or environment variables)
IMAGE_DIR = os.path.join('../raw_data', 'media')
DB_CONNECTION = "postgresql://postgres:5492460@localhost:5432/medical_dw"

# Database setup
engine = create_engine(DB_CONNECTION)
metadata = MetaData()

# Define detection results table
detection_results = Table(
    'detection_results', metadata,
    Column('id', Integer, primary_key=True),
    Column('image_path', String),
    Column('class_label', String),
    Column('confidence', Float),
    Column('x_min', Integer),
    Column('y_min', Integer),
    Column('x_max', Integer),
    Column('y_max', Integer)
)

# Create the table if it doesn't exist
metadata.create_all(engine)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Use YOLOv5 small model

def list_image_files(directory: str) -> List[str]:
    """List all image files in the given directory."""
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    logging.info(f"Found {len(image_files)} images for processing.")
    return image_files

def detect_objects(image_path: str) -> Optional[List[Dict]]:
    """
    Perform object detection on an image using YOLOv5.

    Args:
        image_path (str): Path to the image file.

    Returns:
        Optional[List[Dict]]: List of detection data or None if an error occurs.
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            logging.error(f"Failed to load image: {image_path}")
            return None

        # Perform detection
        results = model(img)
        detections = results.xyxy[0].numpy()  # Get detection results

        # Extract relevant data
        detection_data = []
        for detection in detections:
            x_min, y_min, x_max, y_max, confidence, class_id = detection
            class_label = model.names[int(class_id)]  # Get class label
            detection_data.append({
                'image_path': image_path,
                'class_label': class_label,
                'confidence': float(confidence),
                'x_min': int(x_min),
                'y_min': int(y_min),
                'x_max': int(x_max),
                'y_max': int(y_max)
            })

        logging.info(f"Detected {len(detection_data)} objects in {image_path}")
        return detection_data

    except Exception as e:
        logging.error(f"Error processing {image_path}: {str(e)}")
        return None

def save_to_db(detection_data: List[Dict]) -> None:
    """
    Save detection results to the database.

    Args:
        detection_data (List[Dict]): List of detection data to save.
    """
    try:
        if not detection_data:
            logging.warning("No detection data to save.")
            return

        with engine.connect() as conn:
            for detection in detection_data:
                logging.debug(f"Saving detection: {detection}")
                conn.execute(detection_results.insert().values(**detection))
            conn.commit()  # Explicitly commit the transaction
        logging.info(f"Saved {len(detection_data)} detections to database")
    except SQLAlchemyError as e:
        logging.error(f"Database insertion failed: {str(e)}")

def visualize_detection(image_path: str, detection_data: List[Dict]) -> None:
    """
    Visualize detection results on an image.

    Args:
        image_path (str): Path to the image file.
        detection_data (List[Dict]): List of detection data.
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            logging.error(f"Failed to load image: {image_path}")
            return

        # Convert to RGB for matplotlib
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create plot
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels
        for detection in detection_data:
            x_min, y_min, x_max, y_max = detection['x_min'], detection['y_min'], detection['x_max'], detection['y_max']
            confidence, class_label = detection['confidence'], detection['class_label']

            # Validate bounding box coordinates
            if x_min >= x_max or y_min >= y_max:
                logging.warning(f"Invalid bounding box: {x_min}, {y_min}, {x_max}, {y_max}")
                continue

            # Draw bounding box
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)

            # Add label and confidence
            label = f"{class_label} ({confidence:.2f})"
            ax.text(x_min, y_min - 5, label, color='r', fontsize=12, backgroundcolor='white')

        plt.axis('off')
        plt.show()

    except Exception as e:
        logging.error(f"Error visualizing detection for {image_path}: {str(e)}")

def main():
    """Main function to process images and perform object detection."""
    image_files = list_image_files(IMAGE_DIR)

    # Process all images in the directory
    for image_path in image_files:
        detection_data = detect_objects(image_path)
        if detection_data:
            save_to_db(detection_data)

    # Visualize a sample image
    if image_files:
        sample_image = image_files[50]  # Use the first image for visualization
        sample_detections = detect_objects(sample_image)
        if sample_detections:
            visualize_detection(sample_image, sample_detections)
        else:
            logging.warning("No detections found for the sample image.")

if __name__ == "__main__":
    main()

    # Display logs
    with open('object_detection.log', 'r') as log_file:
        print(log_file.read())