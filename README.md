# Introduction
Skin cancer is a significant global health concern, with early detection being crucial for effective treatment. Deep Learning has emerged as a powerful tool for automating the process of skin cancer diagnosis, offering the potential to improve accuracy and save lives. This project on GitHub aims to demonstrate the application of Deep Learning in skin cancer analysis, providing a comprehensive guide for training and deploying a skin cancer classification model using Python and Jupyter Notebook. The project leverages popular deep learning libraries such as tensorflow, Keras and Jupyter Notebook for a clear and interactive development environment.

Key Components:
1. Data Collection: Explore methods for gathering and organizing a dataset of skin cancer images. High-quality and diverse data are essential for training a robust model.
2. Data Preprocessing: Before feeding the data into the deep learning model, techniques such as data augmentation, resizing, and normalization are required to ensure the data is prepared for effective training.
3. Model Architecture: I'll delve into the architecture of the deep learning model, explaining the choice of convolutional neural networks (CNNs) and how to design an architecture optimized for skin cancer classification.
4. Model Training: Learn how to train the model efficiently, including techniques such as transfer learning and fine-tuning to improve performance. Hyperparameter tuning and optimization are important to optimise the training results.
5. Model Evaluation: Various metrics and techniques will be discussed for evaluating the model's performance, including accuracy, precision, recall, and F1-score.
6. Deployment: Deploy the trained model for practical use, whether through a web application or as part of a larger healthcare system.

# Models

Four CNN architectures are explored for transfer learning:
- InceptionV3 (2015) - Unique inception modules with convolutions of varying sizes help capture multi-scale features. Has 159 layers.
- Xception (2016) - Extends Inception design using depthwise separable convolutions. 126 layers total.
- ResNet50V2 (2016) - 50-layer residual network allowing very deep training. Uses identity shortcut connections.
- DenseNet121 (2017) - Densely connected layers allow feature reuse. Has only 121 layers but high parameter efficiency.

*These models are pretrained on ImageNet and provide a strong starting point.

# Training Methodology
The training workflow involves:

- Fine-tuning the fully connected layers of each base model
  - Adding a global average pooling layer
  - Batch normalization for stability
  - Dense layers with ReLU activation
  - Dropout regularization
  - Output layer with softmax activation
- Data augmentation via flipping, rotating, shifting, etc to expand the training set
- Training with early stopping regularization
- Tracking model loss, accuracy, and recall on a validation set
- Selecting the best epoch with optimal validation metrics
- 
The training data consists of about 10,000 images from the HAM10000 dataset spanning 7 skin cancer classes:
- Actinic keratosis (akiec)
- Basal cell carcinoma (bcc)
- Benign keratosis (bkl)
- Dermatofibroma (df)
- Melanoma (mel)
- Melanocytic nevi (nv)
- Vascular lesions (vasc)

# Requirements
- [Anaconda](https://www.anaconda.com/download/)
- Tensorflow, Numpy, Pandas, Matplotlib Libraries (Installed in the Anaconda cloud environment)
- [Datasets](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) (From Kaggle)
- GPU: Nvidia Geforce RTX 2060 (Used in this project)
