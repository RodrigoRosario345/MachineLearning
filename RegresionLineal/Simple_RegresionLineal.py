import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# lectura de datos

train = pd.read_csv('RegresionLineal/train.csv')
test = pd.read_csv('RegresionLineal/test.csv')

# visualizacion de datos

entrenamiento = train.head(3)

print(entrenamiento)