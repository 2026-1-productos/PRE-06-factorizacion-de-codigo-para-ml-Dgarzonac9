#
# Busque los mejores parametros de un modelo ElasticNet para predecir
# la calidad del vino usando el dataset de calidad del vino tinto de UCI.
#
# Consideere los siguentes valores de los hiperparametros y obtenga el
# mejor modelo.
# (alpha, l1_ratio):
#    (0.5, 0.5), (0.2, 0.2), (0.1, 0.1), (0.1, 0.05), (0.3, 0.2)
#

# importacion de librerias
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .src._internals.calculate_metrics import calculate_metrics
from .src._internals.prepare_data import prepare_data
from .src._internals.print_metrics import print_metrics
from .src._internals.save_model import save_model
from .src._internals.save_model_if_better import save_model_if_better

# preparar los datos
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
x_train, x_test, y_train, y_test = prepare_data(url)

# entrenar el modelo
estimator = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=12345)
estimator.fit(x_train, y_train)

print()
print(estimator, ":", sep="")

# Metricas de error durante entrenamiento
mse, mae, r2 = calculate_metrics(estimator, x_train, y_train)
print_metrics(mse, mae, r2, title="Metricas de entrenamiento:")

# Metricas de error durante testing
mse, mae, r2 = calculate_metrics(estimator, x_test, y_test)
print_metrics(mse, mae, r2, title="Metricas de testing:")
