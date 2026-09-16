# Integrante 2 - Feature Selection con Algoritmo Genético

## Actividad 02 - Machine Learning

Este ejemplo implementa desde cero un algoritmo genético para seleccionar características del dataset **Breast Cancer Wisconsin (Diagnostic)**.

### Archivos
- `01_feature_selection_ga.ipynb`: notebook principal, ya ejecutado.
- `feature_selection_ga.py`: versión en script.
- `Guia_Exposicion_Integrante2.pdf`: guía corta para la defensa oral.

### Dataset
- 569 registros.
- 30 características numéricas originales.
- Problema de clasificación binaria.
- No requiere One-Hot Encoding.

### Modelo
Regresión Logística + StandardScaler.

### Cromosoma
Vector binario de 30 genes:
- `1`: usar la característica.
- `0`: descartar la característica.

### Ciclo del AG
1. Inicialización aleatoria.
2. Evaluación mediante fitness.
3. Selección por torneo.
4. Cruzamiento uniforme.
5. Mutación.
6. Elitismo y reemplazo.
7. Terminación después de 12 generaciones.

### Fitness
Se utiliza accuracy por validación cruzada con una pequeña penalización por número de características.

```text
fitness = accuracy_cv - 0.01 * (n_features / 30)
```

### Ejecución
Notebook:
```bash
jupyter notebook 01_feature_selection_ga.ipynb
```

Script:
```bash
python feature_selection_ga.py
```

### Dependencias
```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

### Idea central para la exposición
El algoritmo genético no modifica la regresión logística. Lo que evoluciona es la máscara que decide qué variables recibe el modelo.
