# Brain Tumor Classification Model using Resnet-50, VGG-16, and ConvXGB

## Overview
This project implements a brain tumor classification model using Resnet-50, VGG-16, and ConvXGB architectures. The goal of this project is to accurately classify brain tumor images into four categories: meningioma, glioma, pituitary tumor, and no tumor.

# Requirements
Python 3.6 or higher
TensorFlow 2.0 or higher
Keras 2.3.1 or higher
Scikit-learn 0.22 or higher
NumPy 1.18 or higher
Pandas 1.0 or higher
Matplotlib 3.1.3 or higher
XGBoost 1.2.0 or higher

## Dataset
The model was trained and tested on the Brain Tumor Classification dataset, which can be obtained from the Kaggle website.

The dataset consists of  Brain Magnetic Resonance Images (MRI). 
The Training folder contains four subfolders: glioma, meningioma, no tumor and pituitary. 

The Training folder consists at total of 5712 images and the Testing folder consists of 1310 images.

Please Click on [Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) to download.


## Model Architectures
This project implements three different architectures for the brain tumor classification model:

- Resnet-50
- VGG-16 
- Convolutional eXtreme Gradient Boosting 

## Results
The models achieved the following results:

- Resnet-50: accuracy of 88.44 %;
- VGG-16: accuracy of 96.80 %;
- ConvXGB: accuracy of 98.58 %;

## Usage
* Open the jupyter notebook and run each cells of the model implemented  (Note: Both the Dataset and the Notebook has to be in the same folder)
* Please click on the links to view the code of each implementation.
- [CXGB](https://github.com/ManojGowda27/Brain-Tumor-Classification/blob/main/CXGB/CXGBoost.ipynb)
- [ResNet-50](https://github.com/ManojGowda27/Brain-Tumor-Classification/blob/main/ResNet-50/ResNet50.ipynb)
- [VGG-16](https://github.com/ManojGowda27/Brain-Tumor-Classification/blob/main/VGG-16/vgg16.ipynb)
* Once the Jupyter Notebook opens, you can run the Notebook and make appropriate changes.

## Conclusion
- In conclusion, each of the three models (ResNet, Convolutional eXtreme Gradient Boosting, and Visual Geometry Group) has its own strengths and limitations, which should be carefully considered before choosing the appropriate model for a specific application. 
- ResNet is a powerful deep learning model for image classification, but it can be prone to overfitting and requires significant computational resources. 
- Convolutional eXtreme Gradient Boosting is a complex model that requires careful hyperparameter tuning, but it is effective for handling complex or heterogeneous data. 
- VGG is a popular and effective deep learning model for image classification, but it may not be suitable for all types of problems and suffers from the vanishing gradient problem. 
- Ultimately, the choice of model depends on the nature of the problem, the available resources, and the desired performance metrics.

