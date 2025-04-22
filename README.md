# **Wine Classification Using Machine Learning 🍷🤖**

This project applies multiple machine learning models to classify wine types based on various physicochemical features. The dataset includes features such as alcohol content, acidity, phenols, and other chemical properties, which are used to predict the wine type.

## **Project Overview 📝**

The goal of this project is to explore different machine learning algorithms for predicting the type of wine. The models used include:

- **K-Nearest Neighbors (KNN)** 🔍
- **Naive Bayes** 📉
- **Decision Trees** 🌳
- **Multiple Linear Regression** 📊
- **Neural Networks** 🧠

These models are evaluated using common metrics such as precision, recall, F1 score, accuracy, and error rate to assess their effectiveness in predicting wine types.

## **Dataset 📦**

The dataset used in this project is the well-known **Wine dataset**, which contains data on the chemical properties of wine, including:

- **Alcohol** 🍇
- **Malic Acid** 🍋
- **Ash** 🧂
- **Alcalinity** ⚗️
- **Magnesium** 🧪
- **Phenols** 🍷
- **Flavanoids** 🍃
- **Nonflavonoids** 🍂
- **Proanthocyanins** 🍒
- **Color Intensity** 🌈
- **Hue** 🌟
- **Dilution** 💧
- **Proline** 💎

The dataset is split into a **training set** (70%) and a **testing set** (30%) for evaluating the models.

## **Algorithms Implemented 🔧**

### **1. K-Nearest Neighbors (KNN) 🧑‍🤝‍🧑**

KNN is used to classify wines based on the similarity of their features to those in the training data. The model predicts the wine type by finding the most similar data points in the feature space.

### **2. Naive Bayes 📉**

The Naive Bayes classifier uses probability to classify wine types based on the features in the dataset. It assumes independence between the features, which simplifies the computation.

### **3. Decision Trees 🌳**

A decision tree model is trained to classify wine types by learning decision rules from the dataset. The tree is built using the features of the dataset and is visualized for better interpretability.

### **4. Multiple Linear Regression 📊**

Linear regression is used to predict the alcohol content based on other features of the dataset, helping us understand how different physicochemical properties relate to each other.

### **5. Neural Networks 🧠**

A neural network is trained to predict wine types using the dataset's features, offering a flexible and powerful method for classification.

## **Evaluation Metrics 📈**

Each model is evaluated using the following metrics:

- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**: The ratio of correctly predicted positive observations to all observations in the actual class.
- **F1 Score**: The weighted average of precision and recall, providing a balance between the two.
- **Accuracy**: The percentage of correct predictions in the total dataset.
- **Error Rate**: The percentage of incorrect predictions.

## **Technologies Used 🛠️**

- **R**: Programming language for data analysis and machine learning.
- **Libraries**:
  - `caret` for model training and evaluation.
  - `e1071` for Naive Bayes implementation.
  - `class` for KNN.
  - `rpart`, `rpart.plot` for decision trees.
  - `neuralnet` for neural networks.
  - `ggplot2`, `corrplot` for data visualization.

Results 🏆
After running the models, you will obtain performance metrics for each algorithm, including precision, recall, F1 score, accuracy, and error rate. These results will help in comparing the effectiveness of the different models for wine classification.

Conclusion 🎯
This project demonstrates the application of multiple machine learning models to solve the problem of wine classification. By comparing the models' performance, we can determine which algorithm works best for this type of dataset.

License 📜
This project is licensed under the MIT License - see the LICENSE file for details.
