# Predicción de generación de energía con turbinas

Este proyecto entrena y compara modelos de regresión para predecir la potencia
continua `MW_ST` de un proceso de generación de energía con turbinas.

El entrenamiento respeta el orden cronológico de las observaciones: utiliza el
80 % inicial para entrenar y validar los modelos y reserva el 20 % más reciente
como conjunto de prueba. De esta forma, la evaluación representa mejor el uso
del modelo para predecir períodos futuros.

## Requisitos

- Python 3.10 o posterior.
- Un archivo privado `dataset_CC02.csv` con las columnas indicadas más abajo.

Las dependencias principales son:

- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- joblib

## Instalación

Desde la carpeta del proyecto, crear y activar un entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instalar las librerías:

```powershell
python -m pip install --upgrade pip
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
```

En Linux o macOS, la activación equivalente es:

```bash
source .venv/bin/activate
```

## Dataset confidencial

El archivo `dataset_CC02.csv` contiene información confidencial y **no se
incluye en el repositorio**. Tanto el dataset como el entorno virtual están
declarados en `.gitignore`:

```gitignore
.venv
dataset_CC02.csv
```

Cada usuario autorizado debe colocar su copia del archivo en la misma carpeta
que `train_turbines_model.py`. No se debe quitar esta regla de `.gitignore`,
subir el dataset a Git ni incluir muestras con información real en incidencias,
commits o documentación.

El CSV debe contener 10 columnas, sin datos faltantes:

| Columna | Uso |
|---|---|
| `fecha` | Timestamp usado para ordenar cronológicamente |
| `MW_ST` | Variable objetivo continua, expresada en MW |
| `MW_TG` | Variable de estado |
| `Humedad` | Perturbación |
| `Presion` | Perturbación |
| `Temp` | Perturbación |
| `Viento` | Perturbación |
| `Viento_cos` | Componente transformada del viento |
| `Viento_sin` | Componente transformada del viento |
| `VIGV` | Perturbación |

`fecha` debe poder convertirse a fecha y hora. Las otras nueve columnas deben
ser numéricas. El script detiene la ejecución si encuentra columnas ausentes,
fechas inválidas, timestamps duplicados o valores faltantes.

## Entrenamiento

Con el entorno virtual activo y el dataset en la carpeta del proyecto:

```powershell
python train_turbines_model.py
```

El script realiza lo siguiente:

1. Carga y valida el dataset.
2. Ordena las filas por `fecha`.
3. Reserva el 20 % más reciente para la prueba final.
4. Compara ocho alternativas mediante cinco particiones de
   `TimeSeriesSplit`.
5. Selecciona el modelo con menor MAE medio de validación.
6. Evalúa el ganador sobre el período de prueba.
7. Reentrena el modelo ganador con todo el dataset y lo guarda.

Los modelos comparados son un baseline por mediana, regresión lineal, Random
Forest, Gradient Boosting, XGBoost, SVR con kernel RBF, K-Nearest Neighbors y
una red neuronal MLP. El escalado se ejecuta dentro de un `Pipeline` para los
modelos que lo requieren, evitando fuga de información entre particiones.

## Métricas

- **MAE (MW):** criterio principal de selección. Indica el error absoluto medio
  en la misma unidad que la potencia.
- **RMSE (MW):** penaliza con mayor intensidad los errores grandes.
- **R²:** indica qué proporción de la variabilidad explica el modelo. Cuanto más
  cercano a 1, mejor.

## Resultados obtenidos

La ejecución sobre las 5.268 observaciones disponibles produjo esta comparación
en validación temporal:

| Modelo | MAE (MW) | RMSE (MW) | R² |
|---|---:|---:|---:|
| Random Forest | 0,7528 | 1,0665 | 0,5700 |
| Gradient Boosting | 0,7650 | 1,1086 | 0,5391 |
| Regresión lineal | 0,7694 | 0,9971 | 0,5857 |
| XGBoost | 0,8374 | 1,1905 | 0,4731 |
| K-Nearest Neighbors | 0,9144 | 1,3351 | 0,3097 |
| SVR (RBF) | 1,0092 | 1,4572 | 0,2363 |
| Baseline (mediana) | 1,6738 | 2,3471 | -0,8623 |
| Red neuronal (MLP) | 2,3386 | 3,5093 | -6,3811 |

Random Forest obtuvo el menor MAE de validación. Sobre las 1.054 observaciones
más recientes, desde el 24 de julio de 2021 a las 15:00, consiguió:

- MAE: **0,7202 MW**
- RMSE: **0,8936 MW**
- R²: **0,8587**

Estos resultados corresponden al dataset y a la división temporal actuales; al
cambiar o ampliar los datos, el modelo ganador y sus métricas pueden variar.

## Archivos generados

Después del entrenamiento se crean:

- `best_turbines_model.joblib`: pipeline ganador reentrenado con todos los
  datos, junto con metadatos y métricas.
- `resultados_modelos.csv`: comparación completa de validación.
- `comparacion_modelos.png`: MAE por modelo, serie real frente a predicción y
  gráfico de residuos.

## Realizar una predicción

El archivo `.joblib` contiene un diccionario. El pipeline entrenado está en la
clave `model` y el orden esperado de las variables está en `features`:

```python
import joblib
import pandas as pd

artefacto = joblib.load("best_turbines_model.joblib")

nueva_observacion = pd.DataFrame(
    [
        {
            "MW_TG": 175.0,
            "Humedad": 30.0,
            "Presion": 1005.0,
            "Temp": 20.0,
            "Viento": 10.0,
            "Viento_cos": 0.5,
            "Viento_sin": 0.866,
            "VIGV": -0.1,
        }
    ],
    columns=artefacto["features"],
)

prediccion_mw = artefacto["model"].predict(nueva_observacion)[0]
print(f"MW_ST estimado: {prediccion_mw:.3f} MW")
```

Los números del ejemplo son ilustrativos y no representan datos reales de la
planta. Para obtener una predicción válida, deben utilizarse mediciones con las
mismas unidades, transformaciones y significado que las usadas durante el
entrenamiento.
