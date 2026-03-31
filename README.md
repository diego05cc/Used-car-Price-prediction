# Used Car Price Prediction Web Application

## Project Description

This project consists of a web-based application developed using Flask that demonstrates the application of Machine Learning techniques to predict the price of used cars. The system implements a linear regression model trained on real-world data to estimate vehicle prices based on relevant features such as year and mileage.

Additionally, the application includes several Machine Learning use cases and conceptual explanations, providing a comprehensive understanding of how these techniques can be applied in real-world scenarios.

In the extended version of the application, classification models such as Logistic Regression and Decision Tree have been integrated. These models allow the system to classify vehicles into categories (e.g., high price or low price) and provide a deeper understanding of supervised learning techniques.

---

## Objectives

* Develop a functional web application using Flask
* Implement a linear regression model for prediction
* Analyze and preprocess a real dataset
* Provide an interactive interface for user input and prediction
* Demonstrate practical Machine Learning use cases
* Implement classification models such as Logistic Regression and Decision Tree
* Evaluate classification models using standard metrics

---

## Machine Learning Model

The model used in this project is based on Linear Regression, a supervised learning algorithm designed to predict continuous numerical values.

Input variables:

* Year
* Mileage

Output:

* Predicted car price

### Additional Models

In addition to Linear Regression, the application includes the following classification models:

#### Logistic Regression

Logistic Regression is used for binary classification tasks. In this project, it is used to classify cars into categories such as high price or low price based on their features.

#### Decision Tree

Decision Tree is a supervised learning algorithm that splits data into branches to make predictions. It is useful for handling non-linear relationships and is easy to interpret.

---

## Dataset

The dataset used in this project was obtained from a public repository such as Kaggle. It contains information about used vehicles, including attributes such as year, mileage, engine size, and price.

This data was used to train the regression model and identify relationships between variables.

For classification tasks, the dataset was adapted by creating a target variable that categorizes cars into different classes based on price thresholds.

---

## Model Evaluation

For classification models, the following evaluation metrics were implemented:

* Confusion Matrix
* Accuracy
* Precision
* Recall
* F1-Score
* ROC Curve and AUC

These metrics allow a better understanding of model performance and prediction quality.

---

## Application Features

* Interactive web interface built with Flask
* Linear Regression prediction system
* Logistic Regression classification module
* Decision Tree classification module
* Visualization of evaluation metrics (confusion matrix and ROC curve)
* Multiple Machine Learning use cases
* Clean and responsive user interface

---

## Project Structure

```
Used-car-Price-prediction/
│── app.py
│── model.py
│── model.pkl
│── logistic_model.pkl
│── tree_model.pkl
│── dataset.csv
│── static/
│   ├── style.css
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── tree_confusion_matrix.png
│── templates/
│   ├── index.html
│   ├── regression.html
│   ├── result.html
│   ├── logistic.html
│   ├── logistic_app.html
│   ├── logistic_result.html
│   ├── tree.html
│   ├── tree_app.html
│   ├── tree_result.html
│   ├── metrics.html
│   ├── tree_metrics.html
```

---

## Git and Branch Management

The project was developed using Git and GitHub with proper version control practices.

Branches used:

* main
* logistic-regression
* decision-tree

Each branch was used to implement specific features and later merged into the main branch, demonstrating a structured development workflow.

---

## Technologies Used

* Python
* Flask
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn
* HTML and CSS

---

## How to Run the Project

1. Clone the repository:

```
git clone https://github.com/your-username/Used-car-Price-prediction.git
```

2. Navigate to the project folder:

```
cd Used-car-Price-prediction
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the application:

```
python app.py
```

5. Open in browser:

```
http://127.0.0.1:5000
```

---

## Conclusions

This project demonstrates how Machine Learning models can be integrated into a web application to solve real-world problems. It highlights the differences between regression and classification models and shows how different algorithms can be applied depending on the problem type.

Additionally, it reinforces the importance of model evaluation, data preprocessing, and user interaction in the development of intelligent systems.

---

## Future Improvements

* Add more features to improve prediction accuracy
* Use more advanced models such as Random Forest
* Improve UI/UX design
* Deploy the application to a cloud platform

