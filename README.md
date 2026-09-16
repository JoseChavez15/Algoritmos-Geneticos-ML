# Actividad 02: Algoritmos Genéticos en Machine Learning

**Asignatura:** Aprendizaje de Máquina (IX Ciclo - Grupo B)
**Institución:** Universidad Nacional del Altiplano - Puno
**Fecha de Entrega:** 16/09/2026

## Objetivo
Comprender y aplicar la metaheurística de Algoritmos Genéticos (AG) para resolver tres problemas clave en Machine Learning:
1. **Feature Selection:** Búsqueda de subconjuntos óptimos de características.
2. **Hyperparameter Optimization:** Optimización de hiperparámetros en Random Forest.
3. **Neuroevolution:** Búsqueda de arquitecturas en Redes Neuronales (MLPClassifier).

## Dataset Base
* **Nombre:** Breast Cancer Wisconsin (Diagnostic) [`scikit-learn`]
* **Dimensiones:** 569 muestras, 30 características numéricas.
* **División:** Split estratificado 80% Entrenamiento y 20% Prueba (`random_state=42`).

## Estructura del Repositorio
```text
Actividad02-Algoritmos-Geneticos-ML/
│
├── README.md                          # Documentación principal del repositorio
├── requirements.txt                    # Lista de librerías y dependencias
├── ga_utils.py                         # Módulo compartido de carga y división de datos
│
├── notebooks/
│   ├── 01_feature_selection.ipynb     # Ejemplo 1 (Integrante 2)
│   ├── 02_hyperparameter_opt.ipynb    # Ejemplo 2 (Integrante 3)
│   └── 03_neuroevolution.ipynb        # Ejemplo 3 (Integrante 4)
│
└── docs/
    └── Resumen_Ejecutivo_GrupoB.pdf   # Resumen ejecutivo en PDF (Máximo 2 páginas)
```

## Instalación y Ejecución
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/JoseChavez15/Algoritmos-Geneticos-ML.git
   cd Algoritmos-Geneticos-ML
   ```
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecutar los cuadernos ubicados dentro de la carpeta `notebooks/`.

