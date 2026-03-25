import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Crear datos de ejemplo para autos usados
np.random.seed(42)
n_samples = 1000

# Generar datos sintéticos
years = np.random.randint(2000, 2024, n_samples)
mileages = np.random.randint(10000, 200000, n_samples)

# Precio base disminuye con el año y aumenta con el kilometraje
base_price = 30000
prices = base_price - (2024 - years) * 1000 - (mileages / 1000) * 50
prices += np.random.normal(0, 2000, n_samples)  # ruido
prices = np.maximum(prices, 5000)  # precio mínimo

# Crear DataFrame
data = pd.DataFrame({
    'year': years,
    'mileage': mileages,
    'price': prices
})

# Guardar dataset
data.to_csv('dataset.csv', index=False)

# Entrenar modelo
X = data[['year', 'mileage']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Guardar modelo
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Modelo entrenado y guardado exitosamente!")
print(f"Score en entrenamiento: {model.score(X_train, y_train):.3f}")
print(f"Score en prueba: {model.score(X_test, y_test):.3f}")