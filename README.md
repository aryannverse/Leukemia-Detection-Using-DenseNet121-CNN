# 🩸 Leukemia Detection using DenseNet121 CNN
### *Acute Lymphoblastic Leukemia (ALL) Detection System*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.1-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-API-red?style=for-the-badge&logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39.0-FF4B4B?style=for-the-badge&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

---

## 📊 Project Overview

This project utilizes Deep Learning and Transfer Learning techniques to classify white blood cells from microscopic images. The goal is to accurately distinguish between normal cells and those affected by **Acute Lymphoblastic Leukemia (ALL)** using the C-NMC dataset.

Acute lymphoblastic leukemia (ALL) is the most common type of childhood cancer and accounts for approximately 25% of pediatric cancers. These cells have been segmented from microscopic images and are representative of images in the real world. The task of identifying immature leukemic blasts from normal cells under the microscope is challenging due to morphological similarity; therefore, the ground truth labels were annotated by an expert oncologist.

<img width="1917" height="958" alt="image" src="https://github.com/user-attachments/assets/8149fafc-6af8-4797-b32a-0ce5bd8d8995" />

---

## 🧠 The Architecture: DenseNet121

We employ **DenseNet121**, a densely connected convolutional network architecture where each layer connects to every other layer in a feed-forward fashion. This mitigates the vanishing-gradient problem, strengthens feature propagation, and substantially reduces the number of parameters.

### 🏗️ Model Structure
| Stage | Component | Description |
| :--- | :--- | :--- |
| **1. Base** | **DenseNet121** | Pre-trained on ImageNet with Global Average Pooling. Used for feature extraction (Top layers removed, weights frozen). |
| **2. Norm** | **Batch Normalization** | Stabilizes learning by normalizing inputs (axis=-1, momentum=0.99, epsilon=0.001). |
| **3. Dense** | **Fully Connected (256)** | Custom classification head with ReLU activation and L2 Regularization. |
| **4. Dropout** | **Dropout (0.40)** | Randomly sets 40% of neurons to 0 to prevent overfitting. |
| **5. Output** | **Dense (2)** | Softmax layer for binary classification probabilities (ALL vs. HEM). |


---

## 🧪 Techniques & Mathematics

### 1. Dense Connectivity (DenseNet)
Unlike traditional sequential networks, DenseNet connects every layer to every subsequent layer. This improves information flow and gradients throughout the network. 
$$x_\ell = H_\ell([x_0, x_1, ..., x_{\ell-1}])$$
Where $x_\ell$ receives the concatenated feature-maps of all preceding layers.

### 2. Regularization (L2 Ridge)
To combat overfitting in medical imaging, we apply L2 regularization to the dense layer weights.
* **L2 (Ridge):** Prevents large weights by penalizing the squared magnitude of coefficients.
    $$Loss_{L2} = \lambda \sum w_i^2$$ *(where λ = 1e-4)*

### 3. Optimization (Adam)
We utilize the **Adam** optimizer (Adaptive Moment Estimation) starting with a learning rate of 0.0001. It computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients.

### 4. Custom Callback Strategy
A custom callback loop monitors the validation metrics to implement:
1.  **Learning Rate Scheduling:** Decays the learning rate by a factor of 0.5 if accuracy plateaus (with a patience of 1).
2.  **Early Stopping:** Halts training after 3 learning rate adjustments without improvement to save computational resources.
3.  **Model Checkpointing:** Restores the weights from the epoch with the highest validation performance.

<img width="1638" height="712" alt="clipboard5" src="https://github.com/user-attachments/assets/42940cc3-e589-4bcb-8acf-ec59a8c19ef6" />

---

## 📈 Model Evaluation & Metrics

To assess the performance and generalization capability of the DenseNet121 model, we analyzed both the training history and the final predictions on the unseen Test Set.


### Confusion Matrix
The confusion matrix provides a detailed breakdown of the model's true vs. predicted classifications on the testing data (1600 images). 

<img width="962" height="978" alt="clipboard2" src="https://github.com/user-attachments/assets/12657c19-018a-4c51-ab67-6291ac50e03d" />


### 3. Classification Report Summary
Based on the evaluation of the test set, the model demonstrates strong discriminatory power between **ALL (Leukemia)** and **HEM (Normal)** cells.

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **ALL (Malignant)** | *0.96* | *0.98* | *0.97* | 1091 |
| **HEM (Normal)** | *0.95* | *0.90* | *0.93* | 509 |
| **Overall Accuracy** | | | **95.5%** | **1600** |


---

## 📂 The Dataset

The project uses the **C-NMC (Classification of Normal vs Malignant Cells)** Leukemia dataset.

* **Source:** [Kaggle - Leukemia Classification](https://www.kaggle.com/datasets/andrewmvd/leukemia-classification)
* **Classes:** 1.  `Hem` (Normal)
    2.  `All` (Malignant / Leukemia)
* **Split Configuration:**
    * Training: 70%
    * Validation: 15%
    * Test: 15% (1600 images evaluated in the dashboard: **1091 ALL, 509 HEM**)

---

## 🚀 Installation & Cloning

Follow these steps to get the project running on your local machine.

### 1. Clone the Repository
Open your terminal or command prompt and run:

```bash
# Clone the project
git clone https://github.com/aryannverse/Leukemia-Detection-Using-DenseNet121-CNN.git
```

### 2. Install Dependencies
Install the required libraries listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 3. Download Data
1.  Download the dataset from the [Kaggle Link](https://www.kaggle.com/datasets/andrewmvd/leukemia-classification).
2.  Extract the downloaded folder.
3.  **Important:** Ensure the path in the notebook matches your local data location:
    ```python
    data_dir = 'C-NMC_Leukemia/training_data'
    ```

### 4. Run the Notebook
Launch Jupyter to view and run the training process.

```bash
jupyter notebook Leukemia_Classification.ipynb
```

### 5. Run Streamlit app for metrics and live prediction.
Either train your own model or use the pretrained model in the repo and run the streamlit app using:
```bash
streamlit run leukemia_app.py
```
