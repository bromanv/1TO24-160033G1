import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Cargar los datos del banco
bank_data = pd.read_csv('./bank-additional-full.csv', delimiter=';')

# Generar números aleatorios para la columna 'Cedula' y añadirlos al DataFrame
cedula_column = np.random.randint(1000000000, 9999999999, size=len(bank_data))
bank_data.insert(0, 'CEDULA', cedula_column)

# Convertir variables categóricas a numéricas, excluyendo la variable objetivo y 'CEDULA'
label_encoders = {}
for column in bank_data.columns:
    if column not in ['y', 'CEDULA'] and bank_data[column].dtype == 'object':
        le = LabelEncoder()
        bank_data[column] = le.fit_transform(bank_data[column].astype(str))
        label_encoders[column] = le

# Codificar la variable objetivo ('y') como binaria
bank_data['y'] = bank_data['y'].map({'yes': 1, 'no': 0})

# Separar las características (X) y la variable objetivo (y)
X_bank = bank_data.drop(['y', 'CEDULA'], axis=1)
y_bank = bank_data['y']

# Utilizar RandomForestClassifier para la selección de características importantes
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_bank, y_bank)

# Seleccionar las características más importantes
selector = SelectFromModel(rf, prefit=True)
X_important = selector.transform(X_bank)
selected_features = X_bank.columns[(selector.get_support())]

# Cargar el conjunto de datos de clientes
clientes_data = pd.read_csv('./clientes.csv', delimiter=',')

# Asegurar que clientes_data tenga las mismas columnas que X_bank
clientes_data = clientes_data.reindex(columns=X_bank.columns, fill_value=0)

# Convertir variables categóricas a numéricas en clientes_data usando los mismos LabelEncoders
for column, le in label_encoders.items():
    if column in clientes_data.columns:
        clientes_data[column] = le.transform(clientes_data[column].astype(str))

# Seleccionar las características importantes para clientes_data
clientes_data_important = selector.transform(clientes_data)

# Clasificar los datos de clientes usando DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_important, y_bank)  # Entrenar con las características importantes de bank_data
predictions = dt.predict(clientes_data_important)

# Definir una función para clasificar a los clientes en categorías basado en reglas
def clasificar_categoria(row):
    if row['age'] > 60 and row['duration'] > 500 and row['euribor3m'] < 2 and row['nr.employed'] < 5000:
        return 'cliente black'
    elif row['age'] > 50 and row['duration'] > 400 and row['euribor3m'] < 3 and row['nr.employed'] < 5100:
        return 'cliente platinum'
    elif row['age'] > 40 and row['duration'] > 300 and row['euribor3m'] < 4 and row['nr.employed'] < 5200:
        return 'cliente gold'
    else:
        return 'cliente regular'

# Aplicar la clasificación de categorías a clientes_data
clientes_data['categoria'] = clientes_data.apply(clasificar_categoria, axis=1)

# Imprimir los resultados de la clasificación en las cuatro categorías
print("Clasificación de clientes en categorías:")
print(clientes_data[['CEDULA', 'age', 'duration', 'euribor3m', 'nr.employed', 'categoria']].head())