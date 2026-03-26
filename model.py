import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


def train_and_save_model(dataset_path='dataset.csv', model_path='model.pkl'):
    # Cargar datos
    data = pd.read_csv(dataset_path)

    if 'year' not in data.columns or 'mileage' not in data.columns or 'price' not in data.columns:
        raise ValueError("El dataset debe contener columnas 'year', 'mileage', 'price'.")

    X = data[['year', 'mileage']]
    y = data['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)
    print(f"Modelo entrenado y guardado en {model_path}")
    print(f"Score entrenamiento: {score_train:.3f}")
    print(f"Score prueba: {score_test:.3f}")


if __name__ == '__main__':
    train_and_save_model()
