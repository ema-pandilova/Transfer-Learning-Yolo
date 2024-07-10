# Transfer learning with Yolo for Object Detection in Remote Sensing

This repository contains notebooks for applying transfer learning techniques with the YOLO (You Only Look Once) model to object detection tasks in remote sensing. The focus is on comparing the performance of YOLO models trained from scratch, pre-trained on COCO, and fine-tuned on the DIOR dataset, as well as evaluating on a ships dataset.

## Project overview
Object detection in remote sensing is a challenging task due to the varied and complex nature of the imagery. Transfer learning, particularly with pre-trained YOLO models, offers a promising approach to improve detection accuracy and efficiency. This project aims to:

- **Compare Training Strategies**: Evaluate the performance of YOLO models trained from scratch versus those pre-trained on the COCO dataset and then fine-tuned on the DIOR dataset.
- **Generalization Capabilities**: Assess how well the models generalize to new, unseen datasets, specifically a custom ships dataset.
- **Performance Metrics**: Analyze various metrics such as precision, recall, mean Average Precision (mAP), and inference time to provide a comprehensive evaluation of the models.
## Datasets
The project utilizes the following datasets:

### DIOR (Dataset for Object Detection in Aerial Images):
  - 20 different classes, ranging from tiny to large-scale objects.
  - 23,463 high-quality images (800 x 800 px) originally split into 50% training and 50% test sets. The images are randomly shuffled.
  - The original annotations are in Pascal VOC format and have been converted to YOLO format for compatibility with the YOLOv8 model.
  - The original splits are available via the following links:
      
    - Train/Val Split: [Google Drive Link](https://drive.google.com/uc?id=1ZHbHDM6hYAEGDC_K5eiW0yF_lzVgpuir)
    - Test Split: [Google Drive Link](https://drive.google.com/uc?id=11SXPqcESez9qTn4Z5Q3v35K9hRwO_epr)
  - In order to ensure balanced representation of all object classes across the training, validation, and test sets, a **stratified split** approach was used in the notebooks.
  - The splits of the labels after the stratified splitting are available in the repository under the `labels/kaggle/working/datasets/labels` directory.

### Ships Dataset: 
- contains images of various types of ships in different environments and from multiple angles.
- images are split into training, validation, and test sets to ensure balanced representation and effective model evaluation.
- the splits are available on Kaggle: [Kaggle link](https://www.kaggle.com/datasets/siddharthkumarsah/ships-in-aerial-images)

## Models
For simple reproduction, we provide the used models int the `models` folder. These files are saved after various epochs and contain the weights of the models:
- `coco-dior-ships-29epochs-best.pt` - Best model after 29 epochs, fine-tuned on COCO, DIOR, and ships dataset.
- `yolo-coco-dior-93epochs-best.pt` - Best model after 93 epochs, fine-tuned on COCO and DIOR datasets.
- `yolo-coco-dior-best.pt` - Best overall model fine-tuned on COCO and DIOR datasets.
- `yolo-coco-dior-ships-93epochs-best.pt` - Best model after 93 epochs, fine-tuned on COCO, DIOR, and ships dataset.
- `yolo-coco-ships-71epochs-best.pt` - Best model after 71 epochs, fine-tuned on COCO and ships dataset.
- `yolo-coco-ships-96epochs-best.pt` - Best model after 96 epochs, fine-tuned on COCO and ships dataset.
## Notebooks
The following Jupyter notebooks are included in this repository to guide you through the experiments:

**1. `01_yolo8-from-scratch-on-dior.ipynb`**:

Train a YOLOv8 model from scratch using the DIOR dataset.

**2. `02_pretrained_yolo8_finetune_on_dior.ipynb`**:

Fine-tune a pre-trained YOLOv8 model (initially trained on COCO) on the DIOR dataset.

**3. `03_pretrained_yolo_with_coco_on_ships.ipynb`**:

Evaluate a YOLOv8 model pre-trained on COCO and fine-tuned on the DIOR dataset using the ships dataset.

**4. `04_coco-dior-yolo-finetune-on-ships.ipynb`**:

Fine-tune a YOLOv8 model pre-trained on both COCO and DIOR datasets specifically on the ships dataset.

## Results

The results from the experiments conducted in the notebooks include detailed comparisons of model performance, specifically focusing on precision, recall, and mean Average Precision (mAP) metrics. The tables below summarize the comparative performance of YOLOv8 models fine-tuned on the DIOR dataset and the ships dataset.

### Comparative Performance of YOLOv8 Models Fine-Tuned on DIOR Dataset

| Model                | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
|----------------------|---------------|------------|-------|----------|
| From Scratch         | 0.819         | 0.636      | 0.714 | 0.486    |
| Pre-Trained on COCO  | 0.848         | 0.685      | 0.760 | 0.532    |

### Comparative Performance of YOLOv8 Models Fine-Tuned on Ships Dataset

| Model                         | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
|-------------------------------|---------------|------------|-------|----------|
| Pre-Trained on COCO + DIOR    | 0.692         | 0.468      | 0.592 | 0.384    |
| Pre-Trained on COCO           | 0.726         | 0.407      | 0.573 | 0.364    |

These results highlight the following observations:
- The YOLOv8 model pre-trained on COCO and fine-tuned on the DIOR dataset outperformed the model trained from scratch in all metrics.
- The model pre-trained on both COCO and DIOR datasets demonstrated better generalization capabilities on the ships dataset compared to the model pre-trained only on COCO.


