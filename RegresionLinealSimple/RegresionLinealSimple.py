# regresion lineal simple

# importar librerias

import numpy as np # libreria para operaciones matematicas
import matplotlib.pyplot as plt # libreria para graficar
import pandas as pd # libreria para importar y manejar datasets y manipularlos

# importar el dataset

dataset = pd.read_csv('RegresionLinealSimple/insurance.csv')
print(dataset)
x = dataset.iloc[:100, 2].values # matriz de variables independientes 
y = dataset.iloc[:100, 6].values # vector de variables dependientes, variable a predecir

x = x.reshape(-1, 1) # convertir el vector x en una matriz de una columna
y = y.reshape(-1, 1) # convertir el vector y en una matriz de una columna
# dividir el dataset en conjunto de entrenamiento y conjunto de testing

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=2/3, random_state=0) # dividir el dataset en conjunto de entrenamiento y conjunto de testing ordenados aleatoriamente con un 20% de datos para testing y 80% para entrenamiento x_train y x_test son las variables independientes y_train y y_test son las variables dependientes

# ajustar la regresion lineal simple con el conjunto de entrenamiento

from sklearn.linear_model import LinearRegression

regression = LinearRegression() # crear objeto de la clase LinearRegression para ajustar la regresion lineal simple

regression.fit(x_train, y_train) # ajustar el objeto LinearRegression a la matriz de variables independientes x_train y el vector de variables dependientes y_train


# predecir el conjunto de test

y_pred = regression.predict(x_test) # predecir el vector de variables dependientes y_pred con el conjunto de testing x_test

print(y_test, " = ", y_pred) # imprimir los valores reales y los valores predichos

# visualizar los resultados del conjunto de entrenamiento

plt.scatter(x_train, y_train, color='red') # graficar los puntos de entrenamiento
plt.plot(x_train, regression.predict(x_train), color='blue') # graficar la recta de regresion lineal simple con los puntos de entrenamiento x_train y las predicciones de y_train

plt.title('Sueldo vs Años de experiencia (Conjunto de entrenamiento)') # titulo del grafico

plt.xlabel('Años de experiencia') # etiqueta del eje x
plt.ylabel('Sueldo (en $)') # etiqueta del eje y

plt.show() # mostrar el grafico


#funcion de hipotesis

