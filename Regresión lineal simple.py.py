#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Regresión lineal simple
"""Supóngase que un analista de deportes quiere saber si existe una relación entre el número de veces que batean los jugadores 
de un equipo de béisbol y el número de runs que consigue. 
En caso de existir y de establecer un modelo, podría predecir el resultado del partido"""


# In[2]:


# Tratamiento de datos
import pandas as pd
import numpy as np


# In[ ]:


# Gráficos
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns


# In[ ]:


# Preprocesado y modelado
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[ ]:


# Configuración matplotlib
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')


# In[ ]:


# Configuración warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Datos
# ===============================================================================
equipos = ["Texas","Boston","Detroit","Kansas","St.","New_S.","New_Y.",
           "Milwaukee","Colorado","Houston","Baltimore","Los_An.","Chicago",
           "Cincinnati","Los_P.","Philadelphia","Chicago","Cleveland","Arizona",
           "Toronto","Minnesota","Florida","Pittsburgh","Oakland","Tampa",
           "Atlanta","Washington","San.F","San.I","Seattle"]
bateos = [5659,  5710, 5563, 5672, 5532, 5600, 5518, 5447, 5544, 5598,
          5585, 5436, 5549, 5612, 5513, 5579, 5502, 5509, 5421, 5559,
          5487, 5508, 5421, 5452, 5436, 5528, 5441, 5486, 5417, 5421]

runs = [855, 875, 787, 730, 762, 718, 867, 721, 735, 615, 708, 644, 654, 735,
        667, 713, 654, 704, 731, 743, 619, 625, 610, 645, 707, 641, 624, 570,
        593, 556]

datos = pd.DataFrame({'equipos': equipos, 'bateos': bateos, 'runs': runs})
datos.head(3)


# In[ ]:


# Gráfico
# ==============================================================================
#Dimensiones fig,ax: eje (a= ancho, b=largo) como quiero que se vea el grafico
fig, ax = plt.subplots(figsize=(6, 3.84))
#Se definen los datos x=predictora y=dependiente c= hace referencia al color "b: blue- g: green - r: red - c: cyan -m: magenta y: yellow -k: black -w: white" 
#kind sirve para definir el tipo de grafico en este caso de dispersi+on(scatter)
datos.plot(
    x    = 'bateos',
    y    = 'runs',
    c    = 'firebrick',
    kind = "scatter",
    ax   = ax
)
ax.set_title('Distribución de bateos y runs');


# In[ ]:


# Correlación lineal entre las dos variables
# ==============================================================================
#(para correr la correlación de pearsonr se llama el directorio datos en donde se encuentran guardadas las listas de bateos y runs)
corr_test = pearsonr(x = datos['bateos'], y =  datos['runs'])

#corr_test(0) muestra coefeciente de correlación y corr_test(1) muestra p-value
print("Coeficiente de correlación de Pearson: ", corr_test[0])
print("P-value: ", corr_test[1])


# In[ ]:


#Scikit-learn
# División de los datos en train(entrenamiento) y test
#https://www.aprendemachinelearning.com/sets-de-entrenamiento-test-validacion-cruzada/
# ==============================================================================
X = datos[['bateos']]
y = datos['runs']

#Divida matrices o matrices en subconjuntos de prueba y tren aleatorio

X_train, X_test, y_train, y_test = train_test_split(
                                        X.values.reshape(-1,1),
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )


# In[ ]:


# Creación del modelo
# ==============================================================================
modelo = LinearRegression()
modelo.fit(X = X_train.reshape(-1, 1), y = y_train)


# In[ ]:


# Información del modelo
# ==============================================================================
print("Intercept:", modelo.intercept_)
print("Coeficiente:", list(zip(X.columns, modelo.coef_.flatten(), )))
print("Coeficiente de determinación R^2:", modelo.score(X, y))


# In[ ]:


#Una vez entrenado el modelo, se evalúa la capacidad predictiva empleando el conjunto de test.
# Error de test del modelo 
# ==============================================================================
predicciones = modelo.predict(X = X_test)
print(predicciones[0:3,])

rmse = mean_squared_error(
        y_true  = y_test,
        y_pred  = predicciones,
        squared = False
       )
print("")
print(f"El error (rmse) de test es: {rmse}")


# In[ ]:


# División de los datos en train y test
# ==============================================================================
X = datos[['bateos']]
y = datos['runs']

X_train, X_test, y_train, y_test = train_test_split(
                                        X.values.reshape(-1,1),
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )


# In[ ]:


# Creación del modelo utilizando matrices como en scikitlearn
# ==============================================================================
# A la matriz de predictores se le tiene que añadir una columna de 1s para el intercept del modelo
X_train = sm.add_constant(X_train, prepend=True)

 #prepend=Anteponer
#prepend : bool... Si es true, la constante está en la primera columna. De lo contrario, se agrega la constante (última columna).
#https://www.statsmodels.org/stable/generated/statsmodels.tools.tools.add_constant.html
modelo = sm.OLS(endog=y_train, exog=X_train,)
#endog: array_like = Variable de respuesta endógena 1-d. La variable dependiente.
#exog :array_like= Una matriz nobs x k donde nobs es el número de observaciones y k es el número de regresores. 
#Una intersección no se incluye de forma predeterminada y debe ser agregada por el usuario.

modelo = modelo.fit() #para datos de entrenamiento
print(modelo.summary())


# In[ ]:


# Intervalos de confianza para los coeficientes del modelo
# ==============================================================================
modelo.conf_int(alpha=0.05)


# In[ ]:


# Predicciones con intervalo de confianza del 95%
# ==============================================================================
predicciones = modelo.get_prediction(exog = X_train).summary_frame(alpha=0.05)
predicciones.head(4)


# In[ ]:


# Predicciones con intervalo de confianza del 95%
# ==============================================================================
predicciones = modelo.get_prediction(exog = X_train).summary_frame(alpha=0.05)
predicciones['x'] = X_train[:, 1]
predicciones['y'] = y_train
predicciones = predicciones.sort_values('x')
print(predicciones)


# In[ ]:


# Gráfico del modelo
# ==============================================================================
fig, ax = plt.subplots(figsize=(6, 3.84))

ax.scatter(predicciones['x'], predicciones['y'], marker='o', color = "gray")
ax.plot(predicciones['x'], predicciones["mean"], linestyle='-', label="OLS")
ax.plot(predicciones['x'], predicciones["mean_ci_lower"], linestyle='--', color='purple', label="95% CI")
ax.plot(predicciones['x'], predicciones["mean_ci_upper"], linestyle='--', color='purple')
ax.fill_between(predicciones['x'], predicciones["mean_ci_lower"], predicciones["mean_ci_upper"], alpha=0.1)
ax.legend();


# In[ ]:


# Error de test del modelo 
# ==============================================================================
X_test = sm.add_constant(X_test, prepend=True)
predicciones = modelo.predict(exog = X_test)
rmse = mean_squared_error(
        y_true  = y_test,
        y_pred  = predicciones,
        squared = False
       )
print("")
print(f"El error (rmse) de test es: {rmse}")


# In[ ]:


#Interpretación
"""La columna (coef) devuelve el valor estimado para los dos parámetros de la ecuación del modelo lineal ( β^0  y  β^1 ) 
que equivalen a la ordenada en el origen (intercept o const) y a la pendiente. Se muestran también los errores estándar, 
el valor del estadístico t y el p-value (dos colas) de cada uno de los dos parámetros. 
Esto permite determinar si los predictores son significativamente distintos de 0, es decir, que tienen importancia en el modelo. Para el modelo generado, tanto la ordenada en el origen como la pendiente son significativas (p-values < 0.05).

El valor de R-squared indica que el modelo es capaz de explicar el 27.1% de la variabilidad observada 
en la variable respuesta (runs). 
Además, el p-value obtenido en el test F (Prob (F-statistic) = 0.00906) indica que sí hay evidencias
de que la varianza explicada por el modelo es superior a la esperada por azar (varianza total).

El modelo lineal generado sigue la ecuación:

runs = -2367.7028 + 0.5529 bateos
 
Por cada unidad que se incrementa el número de bateos, el número de runs aumenta en promedio 0.6305 unidades.

El error de test del modelo es de 59.34. Las predicciones del modelo final se alejan en promedio 59.34 unidades del valor real.



"""

