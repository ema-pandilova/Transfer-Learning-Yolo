# Transfer learning with Yolo for Object Detection in Remote Sensing

This repository contains notebooks for applying transfer learning techniques with the YOLO (You Only Look Once) model to object detection tasks in remote sensing. The focus is on comparing the performance of YOLO models trained from scratch, pre-trained on COCO, and fine-tuned on the DIOR dataset, as well as evaluating on a ships dataset.

## Project overview
Object detection in remote sensing is a challenging task due to the varied and complex nature of the imagery. Transfer learning, particularly with pre-trained YOLO models, offers a promising approach to improve detection accuracy and efficiency. This project aims to:

- **Compare Training Strategies**: Evaluate the performance of YOLO models trained from scratch versus those pre-trained on the COCO dataset and then fine-tuned on the DIOR dataset.
- **Generalization Capabilities**: Assess how well the models generalize to new, unseen datasets, specifically a custom ships dataset.
- **Performance Metrics**: Analyze various metrics such as precision, recall, mean Average Precision (mAP), and inference time to provide a comprehensive evaluation of the models.
## Datasets
The project utilizes the following datasets:

- **DIOR (Dataset for Object Detection in Aerial Images)**: DIOR is a large-scale dataset designed for object detection in aerial images. It includes images from different geographic locations and diverse environmental conditions, covering 20 object categories. This dataset is particularly challenging due to the high variability in object appearance, size, and orientation.
- **COCO (Common Objects in Context)**: COCO is a widely-used dataset in the computer vision community, featuring over 200,000 labeled images with 80 object categories. It is commonly used for pre-training object detection models due to its large scale and diverse set of objects.
- **Ships Dataset**: This is a custom dataset containing images of ships, curated to test the models' ability to generalize to specific object detection tasks outside the DIOR and COCO datasets. It includes various types of ships in different environments and from multiple angles.
## Notebooks
The following Jupyter notebooks are included in this repository to guide you through the experiments:

**1. 01_yolo8-from-scratch-on-dior.ipynb**:

Train a YOLOv8 model from scratch using the DIOR dataset.

**2. 02_pretrained_yolo8_finetune_on_dior.ipynb**:

Fine-tune a pre-trained YOLOv8 model (initially trained on COCO) on the DIOR dataset.

**3. 03_pretrained_yolo_with_coco_on_ships.ipynb**:

Evaluate a YOLOv8 model pre-trained on COCO and fine-tuned on the DIOR dataset using the ships dataset.

**4. 04_coco-dior-yolo-finetune-on-ships.ipynb**:

Fine-tune a YOLOv8 model pre-trained on both COCO and DIOR datasets specifically on the ships dataset.

