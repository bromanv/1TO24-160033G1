# -*- coding: utf-8 -*-
from math import ceil

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from generar_graficos import generar_graficos


def cargar_y_preparar_datos():
    print("Cargar Datos")
    # Cargar los datos del banco
    bank_data = pd.read_csv('./bank-additional-full.csv', delimiter=';')
    cedula_column = np.random.randint(1000000000, 9999999999, size=len(bank_data))
    bank_data.insert(0, 'CEDULA', cedula_column)

    # Convertir variables categoricas a numéricas, excluyendo
    # la variable objetivo y CEDULA
    label_encoders = {}
    for column in bank_data.columns:
        if column not in ['y', 'CEDULA'] and bank_data[column].dtype == 'object':
        #if column in ['euribor3m','nr.employed','duration'] and bank_data[column].dtype == 'object':
            le = LabelEncoder()
            bank_data[column] = le.fit_transform(bank_data[column].astype(str))
            label_encoders[column] = le

    # Codificar la variable objetivo ('y') como binaria
    bank_data['y'] = bank_data['y'].map({'yes': 1, 'no': 0})
    print("# Codificar la variable objetivo ('y') como binaria")
    return bank_data, label_encoders

def seleccionar_caracteristicas_y_predecir(bank_data, label_encoders):
    # Separar las características (X) y la variable objetivo (y)
    X_bank = bank_data.drop(['y', 'CEDULA'], axis=1)
    y_bank = bank_data['y']

    # Utilizar RandomForestClassifier para la selección de características importantes
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_bank, y_bank)

    # Seleccionar las características más importantes
    selector = SelectFromModel(rf, prefit=True)

    # Cargar el conjunto de datos de clientes
    clientes_data = pd.read_csv('./clientes.csv', delimiter=',')
    # Asegurar que clientes_data tenga las columnas 'CEDULA' y las mismas columnas que X_bank
    clientes_data_reindexed = clientes_data.reindex(columns=['CEDULA'] + list(X_bank.columns), fill_value=0)

    # Convertir variables categóricas a numéricas en clientes_data usando los mismos LabelEncoders
    for column, le in label_encoders.items():
        if column in clientes_data_reindexed.columns:
            clientes_data_reindexed[column] = le.transform(clientes_data_reindexed[column].astype(str))

    # Seleccionar las características importantes para clientes_data
    X_clientes_important = selector.transform(clientes_data_reindexed.drop(columns=['CEDULA']))

    # Clasificar los datos de clientes usando DecisionTreeClassifier
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(selector.transform(X_bank), y_bank)  # Entrenar con las características importantes de bank_data
    predictions = dt.predict(X_clientes_important)

    # Agregar predicciones al DataFrame de clientes
    clientes_data_reindexed['prediccion'] = predictions

    return clientes_data_reindexed

def clasificar_categoria(row):
    # Definir pesos para los factores
    peso_duration = 0.0198  # Puntos por segundo
    peso_euribor3m = 10  # Factor de ponderación para Euribor
    peso_nr_employed = 0.0019  # Coeficiente para número de empleados
    
    # Calcular puntuación compuesta
    puntuacion = 0
    puntuacion += row['duration'] * peso_duration
    puntuacion += (5 - row['euribor3m']) * peso_euribor3m
    puntuacion += row['nr.employed'] * peso_nr_employed
    puntuacion += 25 * row['age'] / 60
    puntuacion = ceil(puntuacion)
    # Definir umbrales para cada categoría
    if puntuacion > 44:
        return 'cliente black' #+str(puntuacion)
    elif puntuacion > 37:
        return 'cliente platinum' #+str(puntuacion)
    elif puntuacion > 32:
        return 'cliente gold' #+str(puntuacion)
    else:
        return 'cliente regular' #+str(puntuacion)

def clasificar_clientes():
    print("cargar y preparar Listos")
    bank_data, label_encoders = cargar_y_preparar_datos()
    print("cargar y preparar datos listos")
    clientes_data_con_prediccion = seleccionar_caracteristicas_y_predecir(bank_data, label_encoders)
    # Aplicar la clasificación de categorías a clientes_data
    clientes_data_con_prediccion['categoria'] = clientes_data_con_prediccion.apply(clasificar_categoria, axis=1)
    # Imprimir los resultados de la clasificación en las cuatro categorías
    print("Clasificación de clientes en categorías:")
    # Contar la cantidad de clientes en cada categoría de predicción
    categorias_prediccion_contadas = clientes_data_con_prediccion['categoria'].value_counts()
    generar_graficos(categorias_prediccion_contadas, bank_data)
    
    print(categorias_prediccion_contadas)
    print(clientes_data_con_prediccion[['CEDULA', 'categoria']])    
    

if __name__ == "__main__":
  print("Clasificar.py")
  clasificar_clientes()
