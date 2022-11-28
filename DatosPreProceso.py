# plantilla de pre proceso

# importar librerias

import numpy as np # libreria para operaciones matematicas
import matplotlib.pyplot as plt # libreria para graficar
import pandas as pd # libreria para importar y manejar datasets y manipularlos

# importar el dataset

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values # matriz de variables independientes 
y = dataset.iloc[:, 3].values # vector de variables dependientes, variable a predecir

print(x)
print(y)

# Tratamiento de los datos faltantes
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values= np.NAN, strategy= 'median', fill_value=None, verbose=0, copy=True) # crear objeto de la clase Imputer para reemplazar los valores faltantes por la media de la columna axis = 0 es para que sea por columnas

imputer = imputer.fit(x[:, 1:3]) # ajustar el objeto imputer a la matriz de variables independientes x en las columnas 1 y 2 (no incluye el 3)

x[:, 1:3] = imputer.transform(x[:, 1:3]) # reemplazar los valores faltantes por la media de la columna

# codificar datos categoricos
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from sklearn.compose import ColumnTransformer
    
ColumnTransformer_x = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [0])], remainder='passthrough') # crear objeto de la clase ColumnTransformer para codificar las variables categoricas en variables dummy (0,1) y el remainder es para que no se elimine la columna que no se codifica o se cambie al transformar

x = np.array(ColumnTransformer_x.fit_transform(x), dtype=np.str) # ajustar el objeto ColumnTransformer a la matriz de variables independientes x y convertirlo a un array de numpy de tipo string



LabelEncoder_y = LabelEncoder() # crear objeto de la clase LabelEncoder para codificar las variables categoricas en variables entero (0,1,2,3,4,5,6,7,8,9) en este caso solo hay dos categorias (0,1)

y = LabelEncoder_y.fit_transform(y) # ajustar el objeto LabelEncoder a la matriz de variables dependientes y

# dividir el dataset en conjunto de entrenamiento y conjunto de testing

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2) # dividir el dataset en conjunto de entrenamiento y conjunto de testing ordenados aleatoriamente con un 20% de datos para testing y 80% para entrenamiento x_train y x_test son las variables independientes y_train y y_test son las variables dependientes

print(x_train)

print(x_test)

# Escalado de variables
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler() # crear objeto de la clase StandardScaler para escalar los datos

x_train = sc_x.fit_transform(x_train) # ajustar el objeto StandardScaler a la matriz de variables independientes x_train y transformarla para que los datos esten en el mismo rango

x_test = sc_x.transform(x_test) # transformar la matriz de variables independientes x_test tomando en cuenta que debe tener la misma escala que la matriz de variables independientes x_train para escalar los datos

print(x_train)



