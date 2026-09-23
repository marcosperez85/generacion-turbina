# Predicción de generaciónn de energía con turbinas

Predice `MW_ST` con `MW_TG`, `Humedad`, `Presion`, `Temp`, `Viento`,
`Viento_cos`, `Viento_sin` y `VIGV`. La columna `fecha` determina el orden temporal.

## Instalación y ejecución

Usar un entorno Python compatible con las versiones de `requirements.txt`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python search_random_forest.py
python train_random_forest.py
python -m unittest discover -v
```

Colocar el archivo privado `dataset_CC02.csv` en la carpeta del proyecto.
Está excluido por `.gitignore`: no subirlo a Git ni usar datos reales en tests.
Los tests usan datos sintéticos, unittest (incluido en Python) y bosques pequeños.

## Responsabilidades

- `turbines_common.py`: carga, validación de columnas, fechas, faltantes y
  timestamps duplicados; ordenamiento estable por fecha, corte temporal y métricas.
- `search_random_forest.py`: compara los ocho modelos originales (mediana,
  Random Forest, Gradient Boosting, XGBoost, regresión lineal, SVR, KNN y MLP)
  exclusivamente sobre el 80 % inicial, con cinco particiones de TimeSeriesSplit.
  Guarda `resultados_modelos.csv`. Ejecuta RandomizedSearchCV con las mismas
  distribuciones, 40 combinaciones, semilla 42 y MAPE como objetivo.
  Guarda la configuración completa del mejor bosque en `best_rf_params.json`,
  incluyendo los hiperparámetros seleccionados, la semilla y n_jobs.
- `train_random_forest.py`: lee ese JSON sin repetir la búsqueda. Entrena con
  `int(n * 0.8)` filas iniciales, evalúa sobre las restantes y guarda
  `rf_validation.joblib`. Clona la configuración, entrena con todas las filas
  y guarda `rf_production.joblib`.
- `train_turbines_model.py`: entrada compatible que ejecuta ambas etapas.

Las rutas se resuelven respecto de los scripts. La etapa de entrenamiento
requiere el JSON generado por la búsqueda. No se genera un gráfico: la función
anterior de gráficos no se invocaba. El antiguo `best_turbines_model.joblib`
no se actualiza; los consumidores deben elegir uno de los nuevos artefactos.
Los scripts locales de conversión ONNX que usen el nombre anterior necesitan
recibir explícitamente el nuevo archivo mediante `--model`.

## Artefactos y métricas

Cada joblib contiene `model`, `model_name`, `role`, `features`, `target`,
`trained_from`, `trained_until`, `training_rows`, `hyperparameters`,
`test_metrics`, `metrics_source`, `test_from`, `test_until` y `test_rows`.
Para predecir: cargar con `joblib.load` y usar
`artifact["model"].predict(df[artifact["features"]])`.
Los joblib y el JSON generado están ignorados por Git.

Se conservan MAE en MW, RMSE en MW, R2, MAPE porcentual y `within_1pct`,
el porcentaje de errores absolutos relativos estrictamente menores al 1 %.
La evaluación final divide por el valor absoluto real sin epsilon: los ceros
pueden producir infinito o NaN. La búsqueda conserva el scorer de sklearn,
que sí utiliza epsilon. No se modifican estas definiciones estadísticas.

## Límites de la evaluación temporal

La búsqueda y el ajuste de escaladores (dentro de Pipeline) utilizan solamente
el 80 % inicial. No usar el 20 % final para elegir parámetros o repetir decisiones
hasta mejorar sus métricas. Mantener el mismo dataset entre búsqueda y entrenamiento;
un JSON de otro período podría haber utilizado información del holdout actual.
Verificar que las features están disponibles al predecir y no se hayan calculado
con datos futuros. TimeSeriesSplit conserva su gap original de cero.
El MLP conserva early_stopping con validación interna aleatoria: esa validación
interna no es temporal, aunque nunca accede al holdout externo.

Las métricas de ambos artefactos provienen exclusivamente de `rf_validation`.
El modelo de producción ya vio el 20 % final; esas métricas no constituyen una
evaluación independiente del modelo reentrenado. Evaluarlo requiere datos futuros.

## Conversión del modelo de producción a ONNX

Desde la carpeta del proyecto, con el entorno virtual activo y luego de generar
`rf_production.joblib` con `train_random_forest.py`, ejecutar:

```powershell
python convert_validate_onnx.py --model rf_production.joblib --output rf_production.onnx --report rf_production_report.json
```

El conversor usa por defecto `dataset_CC02.csv` y compara las predicciones de
scikit-learn y ONNX sobre el 20 % final. Genera `rf_production.onnx` y el reporte
`rf_production_report.json`, que incluye las diferencias y la equivalencia de
predicciones. Indicar `--model` es necesario porque el valor predeterminado sigue
siendo el antiguo `best_turbines_model.joblib`.

La advertencia de que el modelo ya vio el período de validación es esperable:
se entrenó con el 100 % de los datos. Estas métricas permiten verificar la
conversión, pero no miden el rendimiento sobre datos nuevos.
