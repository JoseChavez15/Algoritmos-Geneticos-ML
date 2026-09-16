# 🧬 Neuroevolution: Búsqueda de Arquitectura de una Red Neuronal con Algoritmos Genéticos

Este proyecto implementa un **Algoritmo Genético (AG)** para buscar
automáticamente una buena arquitectura de una red neuronal
**MLPClassifier** de `scikit-learn`.

La idea principal es separar dos procesos:

-   **Algoritmo Genético:** decide la arquitectura de la red neuronal.
-   **MLPClassifier:** entrena los pesos de la arquitectura seleccionada
    mediante su entrenamiento normal.

El experimento utiliza el dataset **Breast Cancer Wisconsin
(Diagnostic)** y mantiene un conjunto de prueba independiente para
realizar la evaluación final.

------------------------------------------------------------------------

## 📌 Objetivo

Encontrar, mediante un algoritmo genético, una arquitectura de red
neuronal que obtenga un buen desempeño en el problema de clasificación.

El algoritmo genético busca principalmente:

-   Número de capas ocultas.
-   Número de neuronas por capa.
-   Función de activación.
-   Valor de regularización `alpha`.

El algoritmo **no evoluciona directamente los pesos de la red**. Los
pesos son aprendidos por el `MLPClassifier` durante el entrenamiento.

------------------------------------------------------------------------

## 🧠 Flujo general

``` text
Dataset
   │
   ├── Train (80%) ───────────────┐
   │                              │
   │                         Algoritmo
   │                          Genético
   │                              │
   │                   ┌──────────┴──────────┐
   │                   │                     │
   │              Crear población      Evaluar fitness
   │                   │                     │
   │                   │               Crear y entrenar
   │                   │                    MLP
   │                   │                     │
   │                   │                Accuracy CV
   │                   │                     │
   │                   └──── Selección ◄─────┘
   │                         Cruzamiento
   │                         Mutación
   │                         Elitismo
   │                              │
   │                       Nueva generación
   │                              │
   │                         Repetir 10 veces
   │                              │
   │                       Mejor arquitectura
   │                              │
   └── Test (20%) ────────────────┤
                                  ▼
                           Evaluación final
```

------------------------------------------------------------------------

## 🧬 Representación del cromosoma

Cada individuo representa una posible arquitectura de la red neuronal
mediante cinco genes:

``` text
[n_capas, n1, n2, activacion, alpha]
```

  Gen            Descripción                   Valores posibles
  -------------- ----------------------------- ----------------------------
  `n_capas`      Número de capas ocultas       `1`, `2`
  `n1`           Neuronas de la primera capa   `8`, `16`, `32`, `64`
  `n2`           Neuronas de la segunda capa   `0`, `8`, `16`, `32`
  `activacion`   Función de activación         `relu`, `tanh`, `logistic`
  `alpha`        Regularización L2             `0.0001`, `0.001`, `0.01`

### Ejemplo

``` text
[2, 32, 16, relu, 0.001]
```

Representa una red:

``` text
Entrada
   │
   ▼
32 neuronas
   │
   ▼
16 neuronas
   │
   ▼
Salida

Activación: ReLU
Alpha: 0.001
```

Cuando `n_capas = 1`, el valor de `n2` se establece automáticamente en
`0`.

------------------------------------------------------------------------

## ⚙️ Espacio de búsqueda

El espacio de búsqueda utilizado por el AG es:

``` python
SEARCH_SPACE = {
    'n_capas': [1, 2],
    'n1': [8, 16, 32, 64],
    'n2': [0, 8, 16, 32],
    'activacion': ['relu', 'tanh', 'logistic'],
    'alpha': [1e-4, 1e-3, 1e-2],
}
```

Esto permite explorar diferentes configuraciones de la arquitectura sin
probar manualmente cada combinación.

------------------------------------------------------------------------

## 📊 Función de fitness

La función `calcular_fitness()` determina qué tan buena es una
arquitectura.

El procedimiento es:

1.  Convertir el cromosoma a `hidden_layer_sizes`.
2.  Crear un `Pipeline` con:
    -   `StandardScaler`
    -   `MLPClassifier`
3.  Configurar el MLP según los genes del individuo.
4.  Aplicar validación cruzada de **3 folds** sobre el conjunto de
    entrenamiento.
5.  Calcular la accuracy promedio.
6.  Utilizar esa accuracy como fitness.

Conceptualmente:

``` text
Individuo
    │
    ▼
Arquitectura MLP
    │
    ▼
StandardScaler + MLPClassifier
    │
    ▼
Cross Validation (3 folds)
    │
    ├── Accuracy Fold 1
    ├── Accuracy Fold 2
    └── Accuracy Fold 3
             │
             ▼
       Accuracy promedio
             │
             ▼
          FITNESS
```

Por ejemplo:

``` text
Fold 1 = 0.95
Fold 2 = 0.97
Fold 3 = 0.96

Fitness = (0.95 + 0.97 + 0.96) / 3
        = 0.96
```

Si una configuración genera un error durante el entrenamiento, se le
asigna un fitness de `0.0`.

------------------------------------------------------------------------

## 🧪 Preprocesamiento

El modelo utiliza:

``` python
StandardScaler()
```

antes del `MLPClassifier`.

El escalado permite que las variables tengan una escala comparable y
facilita la convergencia durante el entrenamiento de la red neuronal.

El escalador forma parte del `Pipeline`, por lo que el preprocesamiento
y el modelo se evalúan conjuntamente durante la validación cruzada.

------------------------------------------------------------------------

## 🧬 Operadores genéticos

### 1. Selección por torneo

Se seleccionan aleatoriamente **3 individuos** y se elige el que tenga
el mayor fitness.

``` text
Individuo A → 0.91
Individuo B → 0.96  ← seleccionado
Individuo C → 0.93
```

El individuo con mejor fitness tiene mayor posibilidad de convertirse en
padre.

### 2. Cruzamiento

El hijo recibe cada gen aleatoriamente de uno de sus dos padres.

Ejemplo:

``` text
Padre 1 = [2, 64, 8, relu, 0.01]
Padre 2 = [2, 32, 16, tanh, 0.001]

Hijo    = [2, 64, 16, relu, 0.001]
```

El resultado depende de las elecciones aleatorias realizadas para cada
gen.

### 3. Mutación

Cada gen tiene una probabilidad de mutación de:

``` text
20%
```

Si un gen muta, su valor se reemplaza por otro valor válido del espacio
de búsqueda.

Ejemplo:

``` text
Antes:
[2, 32, 16, relu, 0.001]

Después:
[2, 64, 16, relu, 0.001]
```

### 4. Elitismo

El mejor individuo de cada generación pasa directamente a la siguiente
generación.

Esto evita perder la mejor arquitectura encontrada hasta ese momento.

------------------------------------------------------------------------

## 🔄 Ciclo evolutivo

Los parámetros principales son:

``` python
POP_SIZE = 20
N_GEN = 10
P_MUT = 0.2
```

Por lo tanto:

-   **20 individuos** por población.
-   **10 generaciones**.
-   **20 % de probabilidad de mutación** por gen.

En cada generación se realiza:

``` text
1. Evaluar todos los individuos
2. Obtener sus fitness
3. Encontrar el mejor individuo
4. Aplicar elitismo
5. Seleccionar padres
6. Cruzar padres
7. Mutar hijos
8. Completar la nueva población
9. Repetir
```

------------------------------------------------------------------------

## 📈 Evolución del fitness

El notebook registra el mejor fitness de cada generación:

``` python
historial_mejor_fitness = []
```

Estos valores se utilizan para generar una gráfica de:

``` text
Generación vs. Mejor fitness
```

Gracias al elitismo, el mejor fitness registrado no debería disminuir
entre generaciones.

------------------------------------------------------------------------

## 🏆 Evaluación final

Cuando terminan las generaciones, se toma:

``` python
mejor_individuo_global
```

y se construye nuevamente el modelo utilizando su arquitectura.

Después:

``` python
modelo_final.fit(X_train, y_train)
```

entrena el modelo final utilizando todo el conjunto de entrenamiento.

Finalmente:

``` python
y_pred = modelo_final.predict(X_test)
acc_test = accuracy_score(y_test, y_pred)
```

se calcula la accuracy sobre el conjunto de prueba.

### Importante

El conjunto de `test` representa el **20 % separado inicialmente** y no
participa durante la evolución del algoritmo genético.

Esto permite utilizarlo como evaluación final independiente.

------------------------------------------------------------------------

## 📋 Resultado obtenido

En la ejecución documentada se obtuvo:

  Elemento                      Resultado
  ----------------------------- -----------
  Mejor arquitectura            `(64, 8)`
  Función de activación         `relu`
  `alpha`                       `0.01`
  Mejor fitness en validación   `0.9604`
  Accuracy final en test        `0.9604`

La arquitectura `(64, 8)` representa:

``` text
Entrada
   │
   ▼
64 neuronas
   │
   ▼
8 neuronas
   │
   ▼
Salida
```

con activación `ReLU` y regularización `alpha = 0.01`.

------------------------------------------------------------------------

## 🛠️ Tecnologías utilizadas

-   Python
-   NumPy
-   Matplotlib
-   scikit-learn
-   `MLPClassifier`
-   `StandardScaler`
-   `Pipeline`
-   `cross_val_score`
-   Algoritmos Genéticos
-   Neuroevolución
-   Dataset Breast Cancer Wisconsin (Diagnostic)
-   Google Colab / Jupyter Notebook

------------------------------------------------------------------------

## 📁 Estructura esperada

``` text
Algoritmos-Geneticos-ML/
│
├── 03_neuroevolution.ipynb
├── ga_utils.py
├── README.md
└── ...
```

El archivo `ga_utils.py` contiene la función compartida para cargar el
dataset y generar el mismo split utilizado por los notebooks del equipo.

------------------------------------------------------------------------

## ▶️ Ejecución

### 1. Clonar el repositorio

``` bash
git clone https://github.com/JoseChavez15/Algoritmos-Geneticos-ML.git
cd Algoritmos-Geneticos-ML
```

### 2. Abrir el notebook

Abrir:

``` text
03_neuroevolution.ipynb
```

en Google Colab o Jupyter Notebook.

### 3. Ejecutar las celdas

El notebook:

1.  Carga el dataset.
2.  Inicializa la población.
3.  Evalúa los individuos.
4.  Ejecuta el proceso evolutivo.
5.  Grafica la evolución del fitness.
6.  Obtiene la mejor arquitectura.
7.  Evalúa el modelo final sobre test.
8.  Compara el resultado con un MLP por defecto.

------------------------------------------------------------------------

## 💡 Concepto clave

La idea fundamental de este proyecto puede resumirse así:

> **El algoritmo genético evoluciona las decisiones arquitectónicas de
> la red neuronal; el MLP entrena los pesos de cada arquitectura y su
> desempeño se utiliza como fitness.**

En otras palabras:

``` text
AG
 │
 ├── ¿Cuántas capas?
 ├── ¿Cuántas neuronas?
 ├── ¿Qué activación?
 └── ¿Qué regularización?
          │
          ▼
       MLP
          │
          ▼
   Entrena sus pesos
          │
          ▼
      Accuracy
          │
          ▼
       Fitness
          │
          ▼
    Evolución del AG
```

------------------------------------------------------------------------

## 📚 Contexto académico

Este notebook corresponde a la actividad de **Algoritmos Genéticos en
Machine Learning**, específicamente al enfoque de **Neuroevolution:
búsqueda de arquitectura de red neuronal con AG**.

El objetivo de la implementación es demostrar cómo un algoritmo genético
puede utilizarse como estrategia de búsqueda para seleccionar
automáticamente configuraciones de una red neuronal.
