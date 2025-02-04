# Object Detection Using YOLOv5

This repository contains a Python script for performing object detection on images using the YOLOv5 model. The detected objects are saved to a PostgreSQL database, and the results can be visualized using Matplotlib. The project is designed to process images from a specified directory, detect objects using YOLOv5, and store the results in a structured database for further analysis.



## Table of Contents
1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Database Schema](#database-schema)
6. [Visualization](#visualization)
7. [Logs and Monitoring](#logs-and-monitoring)
8. [Next Steps](#next-steps)
9. [Contributing](#contributing)
10. [License](#license)



## Features
- **Object Detection**: Uses YOLOv5 to detect objects in images.
- **Database Integration**: Saves detection results (class label, confidence, bounding box coordinates) to a PostgreSQL database.
- **Visualization**: Visualizes detected objects with bounding boxes and labels on the images.
- **Logging**: Detailed logging for monitoring and debugging.
- **Modular Code**: Well-structured and modular code for easy maintenance and extension.



## Requirements
- Python 3.8 or higher
- PostgreSQL database
- Required Python libraries:
  - `opencv-python`
  - `torch`
  - `torchvision`
  - `sqlalchemy`
  - `matplotlib`



## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Azazh/Object-Detection-Using-YOLO5.git
   cd object-detection-yolo
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up PostgreSQL Database**:
   - Create a database named `medical_dw` (or any name you prefer).
   - Update the `DB_CONNECTION` string in the script with your database credentials:
     ```python
     DB_CONNECTION = "postgresql://username:password@localhost:5432/medical_dw"
     ```

5. **Download YOLOv5 Model**:
   The script automatically downloads the YOLOv5 model using `torch.hub`. Ensure you have an active internet connection during the first run.



## Usage

1. **Prepare Images**:
   - Place your images in the `../raw_data/media` directory (or update the `IMAGE_DIR` variable in the script).

2. **Run the notebook file**:
   ```bash
   cd notebooks
   python object_detection.ipynb
   ```

3. **View Results**:
   - Detection results are saved in the `detection_results` table in the PostgreSQL database.
   - A sample visualization of the first image is displayed using Matplotlib.



## Database Schema

The detection results are stored in the `detection_results` table with the following schema:

| Column        | Type    | Description                          |
|||--|
| `id`          | Integer | Primary key (auto-increment).        |
| `image_path`  | String  | Path to the image file.              |
| `class_label` | String  | Detected object class label.         |
| `confidence`  | Float   | Confidence score of the detection.   |
| `x_min`       | Integer | Bounding box top-left x-coordinate.  |
| `y_min`       | Integer | Bounding box top-left y-coordinate.  |
| `x_max`       | Integer | Bounding box bottom-right x-coordinate. |
| `y_max`       | Integer | Bounding box bottom-right y-coordinate. |



## Visualization

The script includes a function to visualize the detected objects on the images. It draws bounding boxes and labels (with confidence scores) on the images using Matplotlib.

Example:
![Sample Visualization](sample_visualization.png)



## Logs and Monitoring

The script logs all activities to `object_detection.log`. You can monitor the logs for errors, warnings, and informational messages.

Example log:
```
2023-10-10 12:34:56,789 - INFO - Found 100 images for processing.
2023-10-10 12:35:10,123 - INFO - Detected 3 objects in ../raw_data/media/image1.jpg.
2023-10-10 12:35:15,456 - INFO - Saved 3 detections to database.
```



## Next Steps

- **Fine-tune YOLO**: Train the YOLOv5 model on custom datasets for better accuracy.
- **Extend Visualization**: Add support for visualizing multiple images and saving visualizations to disk.
- **Integrate with Data Warehouse**: Combine detection results with other data sources for comprehensive analysis.
- **Add Unit Tests**: Write unit tests for critical functions like `detect_objects` and `save_to_db`.



## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Submit a pull request.



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



## Acknowledgments
- [YOLOv5 by Ultralytics](https://github.com/ultralytics/yolov5) for the object detection model.
- [SQLAlchemy](https://www.sqlalchemy.org/) for database operations.
- [OpenCV](https://opencv.org/) for image processing.
- [Matplotlib](https://matplotlib.org/) for visualization.



