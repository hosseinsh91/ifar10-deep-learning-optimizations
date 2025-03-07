# Optimizing Deep Learning Models on CIFAR-10

## **📌 Project Overview**
This project applies **advanced deep learning techniques** to improve **image classification accuracy** on the **CIFAR-10 dataset**. The notebook tests multiple **activation functions, weight initializers, optimizers, and regularization techniques** to determine the best approach for training neural networks.

### **🚀 Key Features**
✅ **Batch Normalization** – Improves model stability and speeds up convergence  
✅ **Activation Function Comparisons** – **SELU vs. ReLU**  
✅ **Transfer Learning** – Freezing vs. unfreezing layers for efficient training  
✅ **Optimizer Analysis** – Comparing **Adam, SGD, AdaGrad, Nadam**  
✅ **Regularization Techniques** – L1/L2 Regularization, Dropout, Monte Carlo Dropout  
✅ **Training Metrics Visualization** – Accuracy and loss curves  

---

## **📌 Dataset: CIFAR-10**
- **CIFAR-10** consists of **60,000 color images** in **10 categories**, including:
  - Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- Each image is **32x32 pixels with 3 color channels**.

### **📌 Data Preprocessing**
- The dataset is **normalized** by scaling pixel values to `[0,1]`:
  ```python
  (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
  X_train, X_test = X_train / 255.0, X_test / 255.0
