import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generar_graficos(categorias_prediccion_contadas, bank_data, importancias=None):
    # Gráfico Circular para la Distribución de Categorías
    categorias = categorias_prediccion_contadas.index
    sizes = categorias_prediccion_contadas.values
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=categorias, autopct='%1.1f%%', startangle=140)
    plt.title('Distribución de Clientes por Categoría')
    plt.savefig('./distribucion_clientes_categoria.png')
    plt.close()

    # Gráfico de Barras para la Importancia de las Características (si aplicable)
    if importancias is not None:
        df_importancias = pd.DataFrame({'Característica': bank_data.columns[:-1], 'Importancia': importancias})
        df_importancias = df_importancias.sort_values('Importancia', ascending=False)
        plt.figure(figsize=(10, 8))
        plt.barh(df_importancias['Característica'], df_importancias['Importancia'])
        plt.xlabel('Importancia')
        plt.title('Importancia de las Características en la Predicción')
        plt.gca().invert_yaxis()
        plt.savefig('./importancia_caracteristicas.png')
        plt.close()

    # Histograma de Edad
    plt.figure(figsize=(10, 6))
    plt.hist(bank_data['age'], bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribución de la Edad de los Clientes')
    plt.xlabel('Edad')
    plt.ylabel('Frecuencia')
    plt.savefig('./distribucion_edad_clientes.png')
    plt.close()

    # Mapa de Calor de Correlación
    plt.figure(figsize=(25, 15))
    sns.heatmap(bank_data.corr(), annot=True, cmap='coolwarm')
    plt.title('Mapa de Calor de Correlación de las Variables')
    plt.savefig('./mapa_calor_correlacion.png')
    plt.close()
