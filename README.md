# Predicción de generación de energía con turbinas

Este proyecto entrena, compara y optimiza modelos de regresión para predecir la
potencia continua `MW_ST` de un proceso de generación de energía con turbinas.

El flujo respeta el orden cronológico de las observaciones: utiliza el 80 %
inicial para entrenamiento y validación temporal y reserva el 20 % más reciente
como conjunto de prueba final. De esta forma, la evaluación representa mejor el
uso del modelo para predecir períodos futuros y evita mezclar observaciones
futuras dentro del entrenamiento.

## Requisitos

- Python 3.10 o posterior.
- Un archivo privado `dataset_CC02.csv` con las columnas requeridas por el script.
- Las dependencias indicadas en `requirements.txt`.

## Instalación

Desde la carpeta del proyecto, crear y activar un entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instalar las librerías:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

En Linux o macOS, la activación equivalente es:

```bash
source .venv/bin/activate
```

## Dataset confidencial

El archivo `dataset_CC02.csv` contiene información confidencial y **no se
incluye en el repositorio**.

Cada usuario autorizado debe colocar su copia del archivo en la misma carpeta
que `train_turbines_model.py`. No se debe quitar esta regla de `.gitignore`,
subir el dataset a Git ni incluir muestras con información real en incidencias,
commits o documentación.

El script utiliza las siguientes variables predictoras:

- `MW_TG`
- `Humedad`
- `Presion`
- `Temp`
- `Viento`
- `Viento_cos`
- `Viento_sin`
- `VIGV`

La variable objetivo es `MW_ST` y la columna `fecha` se utiliza para mantener
el orden temporal de los datos.

## Entrenamiento y evaluación

Con el entorno virtual activo y el dataset en la carpeta del proyecto:

```powershell
python train_turbines_model.py
```

El script realiza lo siguiente:

1. Carga el dataset y valida columnas requeridas, fechas, valores faltantes y
   timestamps duplicados.
2. Ordena las observaciones cronológicamente por `fecha`.
3. Reserva el 20 % más reciente como conjunto de prueba final.
4. Compara ocho alternativas de regresión sobre el 80 % de entrenamiento
   mediante cinco particiones de `TimeSeriesSplit`.
5. Calcula MAE, RMSE, MAPE y R² para cada modelo durante la validación temporal.
6. Ordena la comparación de modelos según el menor MAPE medio de validación.
7. Ejecuta una optimización específica de Random Forest mediante
   `RandomizedSearchCV`, utilizando nuevamente `TimeSeriesSplit` y MAPE como
   función objetivo.
8. Evalúa el Random Forest optimizado sobre el 20 % más reciente, que no
   participa de la búsqueda de hiperparámetros.
9. Calcula las métricas finales y el porcentaje de predicciones cuyo error
   porcentual absoluto es inferior al 1 %.
10. Reentrena la configuración optimizada con todo el histórico disponible y
    guarda el modelo junto con sus metadatos y métricas.

Los modelos comparados son un baseline por mediana, regresión lineal, Random
Forest, Gradient Boosting, XGBoost, SVR con kernel RBF, K-Nearest Neighbors y
una red neuronal MLP. El escalado se ejecuta dentro de un `Pipeline` para los
modelos que lo requieren, evitando fuga de información entre particiones.

## Optimización de Random Forest

Después de la comparación inicial, el script utiliza `RandomizedSearchCV` para
buscar una mejor configuración de Random Forest sin utilizar el conjunto de
prueba final para seleccionar los hiperparámetros.

Actualmente se exploran combinaciones de:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`

La búsqueda prueba 40 combinaciones y evalúa cada una mediante cinco
particiones temporales. La configuración seleccionada es la que obtiene el
menor MAPE medio de validación.

## Métricas

- **MAPE (%):** criterio principal de optimización y comparación con el objetivo
  de error porcentual. Representa el promedio del error porcentual absoluto.
- **MAE (MW):** error absoluto medio expresado en la misma unidad que la potencia.
- **RMSE (MW):** penaliza con mayor intensidad los errores grandes.
- **R²:** mide la proporción de variabilidad explicada por el modelo.
- **Predicciones con error < 1 %:** porcentaje de observaciones del conjunto de
  prueba cuyo error porcentual absoluto individual es inferior al 1 %.

El objetivo de error porcentual se evalúa actualmente como un MAPE inferior al
1 %. El porcentaje de predicciones individuales dentro del 1 % se informa como
una métrica complementaria y no como el criterio principal de selección.

## Archivos generados

Después de la ejecución se generan:

- `best_turbines_model.joblib`: modelo Random Forest optimizado y reentrenado
  con todo el histórico disponible, junto con metadatos y métricas de prueba.
  El archivo se excluye del repositorio mediante `.gitignore` porque es un
  artefacto generado y puede tener un tamaño considerable.
- `resultados_modelos.csv`: comparación de los ocho modelos durante la
  validación temporal, incluyendo MAE, RMSE, MAPE, R² y variabilidad del error.
- `comparacion_modelos.png`: comparación gráfica de los modelos, serie real
  frente a predicción y análisis de residuos. El archivo también puede
  excluirse del repositorio si se considera un artefacto generado.

## Despliegue

El modelo se entrena actualmente con scikit-learn y se guarda localmente en
formato `joblib`. El despliegue en producción todavía no forma parte de este
script.

Como siguiente etapa se evalúa convertir el Random Forest a formato ONNX para
su ejecución en un runtime compatible. Antes de utilizar el modelo convertido
en producción, las predicciones del modelo ONNX deben validarse contra las del
modelo original de scikit-learn para comprobar que la conversión conserva su
comportamiento.
