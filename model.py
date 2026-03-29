import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def train_models(dataset_path='dataset.csv'):

    data = pd.read_csv(dataset_path)

    X_reg = data[['year', 'mileage']]
    y_reg = data['price']

    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)

    pickle.dump(reg_model, open('model.pkl', 'wb'))

    #new part of the classification

    # Create a binary variable for example: expensive or cheap
    data['price_category'] = (data['price'] > data['price'].median()).astype(int)

    X = data[['year', 'mileage']]
    y = data['price_category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Logistic Regression
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)

    pickle.dump(log_model, open('logistic_model.pkl', 'wb'))

    y_pred = log_model.predict(X_test)

    #metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Logistic Regression Metrics:")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)

    # CONFUSION MATRIX
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.savefig('static/confusion_matrix.png')


    #decision tree piece new
    tree_model = DecisionTreeClassifier()
    tree_model.fit(X_train, y_train)

    pickle.dump(tree_model, open('tree_model.pkl', 'wb'))

    y_pred_tree = tree_model.predict(X_test)

    #metrics TREE
    acc_tree = accuracy_score(y_test, y_pred_tree)
    prec_tree = precision_score(y_test, y_pred_tree)
    rec_tree = recall_score(y_test, y_pred_tree)
    f1_tree = f1_score(y_test, y_pred_tree)

    print("Decision Tree Metrics:")
    print("Accuracy:", acc_tree)
    print("Precision:", prec_tree)
    print("Recall:", rec_tree)
    print("F1 Score:", f1_tree)

    # CONFUSION MATRIX TREE
    cm_tree = confusion_matrix(y_test, y_pred_tree)

    plt.figure()
    sns.heatmap(cm_tree, annot=True, fmt='d')
    plt.title("Decision Tree Confusion Matrix")
    plt.savefig('static/tree_confusion_matrix.png')

    #ROC CURVE
    y_prob = log_model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.savefig('static/roc_curve.png')

    return acc, prec, rec, f1

if __name__ == "__main__":
    train_models()