import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Cargar los datos
bank_data = pd.read_csv('./bank-additional-full.csv', delimiter=';')

# Generar números aleatorios para la columna 'Cedula'
cedula_column = np.random.randint(1000000000, 9999999999, size=len(bank_data))

# Insertar la columna 'CEDULA' en la primera posición
bank_data.insert(0, 'CEDULA', cedula_column)

# Convertir variables categóricas a numéricas, excluyendo la variable objetivo
label_encoders = {}
for column in bank_data.select_dtypes(include=['object']).columns:
    if column != 'y' and column != 'CEDULA':  # Excluyendo la variable objetivo
        le = LabelEncoder()
        # Ajuste a las categorías conocidas
        bank_data[column] = le.fit_transform(bank_data[column].fillna('Unknown'))
        label_encoders[column] = le

# Codificar la variable objetivo
bank_data['y'] = bank_data['y'].map({'yes': 1, 'no': 0})

# Separar las características y la variable objetivo
X_bank = bank_data.drop('y', axis=1)
y_bank = bank_data['y']

# Utilizar RandomForestClassifier para la selección de características
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_bank, y_bank)

# Seleccionar las características más importantes
selector = SelectFromModel(rf, prefit=True)
X_important = selector.transform(X_bank)
selected_features = X_bank.columns[(selector.get_support())]

# Cargar el nuevo conjunto de datos
clientes_data = pd.read_csv('./clientes.csv', delimiter=',')

# Asegurar que clientes_data tenga las mismas columnas que bank_data, incluso si faltan algunas
clientes_data = clientes_data.reindex(columns=X_bank.columns, fill_value=0)

# Usar los LabelEncoder guardados para convertir variables categóricas a numéricas
for column, le in label_encoders.items():
    if column in clientes_data.columns:
        # Transformar las categorías conocidas, manejar desconocidas
        clientes_data[column] = clientes_data[column].map(lambda s: le.transform([s])[0] if s in le.classes_ else 0)

# Seleccionar las características importantes usando el selector entrenado previamente
clientes_data_important = selector.transform(clientes_data)

# Clasificar los datos de clientes
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_bank[selected_features], y_bank)  # Asegúrate de entrenar nuevamente el clasificador si es necesario
predictions = dt.predict(clientes_data_important)

# Imprimir las predicciones
print("Predicciones para los clientes en 'clientes.csv':")
# Si deseas más detalle, como mostrar las predicciones junto a alguna otra información del dataset
print(clientes_data)
for CEDULA, prediction in zip(clientes_data['CEDULA'], predictions):
  if(prediction == 1) :
    print(f"Cliente ID: {CEDULA}, Sí")
