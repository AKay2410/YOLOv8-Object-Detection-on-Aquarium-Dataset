# YOLOv8-Object-Detection-on-Aquarium-Dataset

This project demonstrates training a YOLOv8 model on the Aquarium Dataset using a custom Google Colab notebook. The goal is to detect various aquatic species, including fish, jellyfish, penguins, and more.

## Project Structure
- **Code File/**: Contains the main Colab notebook with training code.
- **README.md**: Project overview and usage instructions.

## Getting Started

### Dataset
The dataset can be downloaded from Kaggle:
- [Aquarium Dataset on Kaggle](https://www.kaggle.com/datasets/slavkoprytula/aquarium-data-cots)
  
To use this dataset, follow these steps in the Colab notebook:
1. Upload your `kaggle.json` file to Colab.
2. Use the Kaggle API to download the dataset.

### Colab Notebook
The Colab notebook file is located in `notebooks/YOLOv8_Aquarium_Object_Detection.ipynb`.

This notebook includes:
- Dataset preparation and loading.
- Training a YOLOv8 model on the dataset.
- Model evaluation and visualization of results.

### YOLOv8 Training Steps
1. **Dataset Preparation**: Prepare the dataset in YOLO format (images and labels).
2. **Model Training**: Train the YOLOv8 model using the provided configurations.
3. **Evaluation and Analysis**: Assess model performance using metrics such as mAP and precision.
4. **Visualization**: Display sample predictions on test images.

## Colab Notebook Details
Within the Colab notebook:
- **Data.yaml Configuration**: Adjusted paths to point to dataset locations.
- **Training Hyperparameters**: Configured epochs, batch size, and image size for optimal performance.
- **Data Augmentation**: Options to apply augmentation (if chosen) and rationale for inclusion/exclusion.
  
### Hyperparameters
The main hyperparameters used in the training process include:
- **Epochs**: 30 (you can adjust as needed)
- **Batch Size**: 16
- **Image Size**: 640

## Results and Analysis
Key metrics analyzed after training:
- **Mean Average Precision (mAP)**: Evaluates model precision across classes.
- **Precision and Recall**: Assesses model accuracy and false positives.
- **Loss Metrics**: Training loss, classification loss, and localization loss.

Sample visualizations of predictions can be found in the `images/` folder.

## Usage
To reproduce the results:
1. Open the Colab notebook and upload `kaggle.json`.
2. Follow the setup steps in the notebook to download the dataset and train the model.
3. View the evaluation results and visualize predictions.

## Future Work
Potential improvements could include:
- Experimenting with other YOLOv8 model sizes.
- Using more data augmentations for robustness.
- Hyperparameter tuning to enhance performance.

## References
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Aquarium Dataset on Kaggle](https://www.kaggle.com/datasets/slavkoprytula/aquarium-data-cots)
