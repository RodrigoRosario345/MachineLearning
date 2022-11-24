import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

regr = linear_model.LinearRegression()

dataset = pd.read_csv('RegresionLinealSimple2/movies2.csv')

datos = pd.DataFrame(dataset)

x = datos['movie_facebook_likes'] # matriz de variables independientes

y = datos['imdb_score'] # vector de variables dependientes, variable a predecir

X = x[:, np.newaxis] # selecciona todas las filas y la columna de x con newaxis

print(X)

print(regr.fit(X, y)) # ajustar la regresion lineal con los datos de x e y

print(regr.coef_) # coeficiente de la regresion lineal

m = regr.coef_[0] # coeficiente de la regresion lineal en la posicion 0

b = regr.intercept_ # coeficiente de la regresion lineal de la interseccion pendiente

y_pred = m*X + b # ecuacion de la recta de regresion lineal

print(f'y = {m}*x + {b}') # imprimir la ecuacion de la recta de regresion lineal

print(regr.predict(X[0:5])) # predecir los valores de y con los valores de x

print('el valor de R^2 es: ', r2_score(y, y_pred)) # calcular el valor de R^2 para la regresion lineal