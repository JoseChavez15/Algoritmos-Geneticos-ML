"""
Actividad 02 - Algoritmos Genéticos en Machine Learning
Integrante 2: Feature Selection con Algoritmo Genético

Objetivo:
Seleccionar un subconjunto de las 30 características del dataset
Breast Cancer Wisconsin usando un algoritmo genético implementado
desde cero y evaluar el subconjunto con Regresión Logística.

No usa DEAP, PyGAD ni otra librería de algoritmos genéticos.
"""

import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")

# =========================================================
# 1. CONFIGURACIÓN
# =========================================================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

POP_SIZE = 20
N_GENERATIONS = 12
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.90
ELITE_SIZE = 2
PENALTY = 0.01

# =========================================================
# 2. DATASET
# =========================================================
data = load_breast_cancer(as_frame=True)
X = data.data.copy()
y = data.target.copy()

print("=" * 70)
print("DATASET")
print("=" * 70)
print(f"Registros: {X.shape[0]}")
print(f"Características originales: {X.shape[1]}")
print(f"Clases: {list(data.target_names)}")
print(f"¿Necesita One-Hot Encoding?: No, las 30 variables son numéricas.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEED
)

N_FEATURES = X.shape[1]
MUTATION_RATE = 1 / N_FEATURES

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=SEED
)

def crear_modelo():
    """Pipeline: estandarización + regresión logística."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            random_state=SEED
        ))
    ])

# =========================================================
# 3. MODELO BASE: USA LAS 30 CARACTERÍSTICAS
# =========================================================
baseline_cv_scores = cross_val_score(
    crear_modelo(),
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)

baseline_cv = baseline_cv_scores.mean()

baseline_model = crear_modelo()
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)
baseline_test = accuracy_score(y_test, baseline_pred)

print("\n" + "=" * 70)
print("MODELO BASE")
print("=" * 70)
print(f"Características usadas: {N_FEATURES}")
print(f"Accuracy CV (train): {baseline_cv:.4f}")
print(f"Accuracy test:       {baseline_test:.4f}")

# =========================================================
# 4. REPRESENTACIÓN DEL CROMOSOMA
# =========================================================
# Cada individuo es un vector binario de longitud 30.
# 1 = usar característica
# 0 = no usar característica
#
# Ejemplo:
# [1, 0, 1, 0, ...]
#  ↑     ↑
# usar   usar
#
def crear_individuo():
    individuo = np.random.randint(0, 2, size=N_FEATURES, dtype=int)

    # Evitar un individuo sin ninguna característica
    if individuo.sum() == 0:
        individuo[np.random.randint(N_FEATURES)] = 1

    return individuo

# =========================================================
# 5. FUNCIÓN DE APTITUD (FITNESS)
# =========================================================
def evaluar_fitness(individuo):
    """
    Fitness = accuracy promedio por validación cruzada
              - penalización pequeña por usar demasiadas variables.

    La penalización hace que el AG valore soluciones compactas.
    """
    mascara = individuo.astype(bool)
    cantidad = int(mascara.sum())

    if cantidad == 0:
        return -1.0, 0.0

    X_seleccionado = X_train.iloc[:, mascara]

    accuracy_cv = cross_val_score(
        crear_modelo(),
        X_seleccionado,
        y_train,
        cv=cv,
        scoring="accuracy"
    ).mean()

    fitness = accuracy_cv - PENALTY * (cantidad / N_FEATURES)

    return float(fitness), float(accuracy_cv)

# =========================================================
# 6. SELECCIÓN POR TORNEO
# =========================================================
def seleccion_torneo(poblacion, fitnesses):
    """
    Escoge 3 individuos al azar y devuelve el de mayor fitness.
    """
    candidatos = np.random.choice(
        len(poblacion),
        size=TOURNAMENT_SIZE,
        replace=False
    )

    ganador = max(candidatos, key=lambda i: fitnesses[i][0])
    return poblacion[ganador].copy()

# =========================================================
# 7. CRUZAMIENTO UNIFORME
# =========================================================
def cruzamiento_uniforme(padre1, padre2):
    """
    Cada gen del hijo puede venir del padre 1 o del padre 2.
    """
    if np.random.rand() > CROSSOVER_RATE:
        return padre1.copy(), padre2.copy()

    mascara = np.random.rand(N_FEATURES) < 0.5

    hijo1 = np.where(mascara, padre1, padre2).astype(int)
    hijo2 = np.where(mascara, padre2, padre1).astype(int)

    if hijo1.sum() == 0:
        hijo1[np.random.randint(N_FEATURES)] = 1

    if hijo2.sum() == 0:
        hijo2[np.random.randint(N_FEATURES)] = 1

    return hijo1, hijo2

# =========================================================
# 8. MUTACIÓN
# =========================================================
def mutacion(individuo):
    """
    Cada gen tiene probabilidad 1/30 de cambiar:
    0 -> 1 o 1 -> 0.
    """
    hijo = individuo.copy()

    genes_a_mutar = np.random.rand(N_FEATURES) < MUTATION_RATE
    hijo[genes_a_mutar] = 1 - hijo[genes_a_mutar]

    if hijo.sum() == 0:
        hijo[np.random.randint(N_FEATURES)] = 1

    return hijo

# =========================================================
# 9. EJECUCIÓN DEL ALGORITMO GENÉTICO
# =========================================================
np.random.seed(SEED)
random.seed(SEED)

poblacion = [crear_individuo() for _ in range(POP_SIZE)]

historial = []
mejor_global = None
mejor_fitness_global = -np.inf
mejor_accuracy_cv_global = 0.0

print("\n" + "=" * 70)
print("EVOLUCIÓN DEL ALGORITMO GENÉTICO")
print("=" * 70)

for generacion in range(1, N_GENERATIONS + 1):

    # 1) EVALUACIÓN
    fitnesses = [evaluar_fitness(ind) for ind in poblacion]

    # Ordenar de mayor a menor fitness
    orden = sorted(
        range(POP_SIZE),
        key=lambda i: fitnesses[i][0],
        reverse=True
    )

    mejor_idx = orden[0]
    mejor_individuo_gen = poblacion[mejor_idx]
    mejor_fitness_gen, mejor_accuracy_gen = fitnesses[mejor_idx]

    # Guardar mejor global
    if mejor_fitness_gen > mejor_fitness_global:
        mejor_fitness_global = mejor_fitness_gen
        mejor_accuracy_cv_global = mejor_accuracy_gen
        mejor_global = mejor_individuo_gen.copy()

    historial.append({
        "generacion": generacion,
        "mejor_fitness": mejor_fitness_gen,
        "mejor_accuracy_cv": mejor_accuracy_gen,
        "caracteristicas": int(mejor_individuo_gen.sum()),
        "fitness_promedio": float(np.mean([f[0] for f in fitnesses]))
    })

    print(
        f"Generación {generacion:02d} | "
        f"fitness={mejor_fitness_gen:.4f} | "
        f"accuracy_cv={mejor_accuracy_gen:.4f} | "
        f"features={int(mejor_individuo_gen.sum()):02d}"
    )

    # 2) ELITISMO: conservar los dos mejores
    nueva_poblacion = [
        poblacion[i].copy()
        for i in orden[:ELITE_SIZE]
    ]

    # 3) SELECCIÓN + CRUZAMIENTO + MUTACIÓN
    while len(nueva_poblacion) < POP_SIZE:
        padre1 = seleccion_torneo(poblacion, fitnesses)
        padre2 = seleccion_torneo(poblacion, fitnesses)

        hijo1, hijo2 = cruzamiento_uniforme(padre1, padre2)

        hijo1 = mutacion(hijo1)
        hijo2 = mutacion(hijo2)

        nueva_poblacion.extend([hijo1, hijo2])

    poblacion = nueva_poblacion[:POP_SIZE]

# =========================================================
# 10. MEJORES CARACTERÍSTICAS
# =========================================================
mascara_final = mejor_global.astype(bool)
caracteristicas_seleccionadas = X.columns[mascara_final].tolist()

print("\n" + "=" * 70)
print("MEJOR SOLUCIÓN ENCONTRADA")
print("=" * 70)
print(f"Fitness: {mejor_fitness_global:.4f}")
print(f"Accuracy CV: {mejor_accuracy_cv_global:.4f}")
print(f"Características originales: {N_FEATURES}")
print(f"Características seleccionadas: {len(caracteristicas_seleccionadas)}")
print(
    f"Reducción: "
    f"{(1 - len(caracteristicas_seleccionadas) / N_FEATURES) * 100:.1f}%"
)

print("\nCaracterísticas seleccionadas:")
for i, nombre in enumerate(caracteristicas_seleccionadas, start=1):
    print(f"{i:02d}. {nombre}")

# =========================================================
# 11. EVALUACIÓN FINAL EN TEST
# =========================================================
modelo_final = crear_modelo()
modelo_final.fit(
    X_train[caracteristicas_seleccionadas],
    y_train
)

pred_final = modelo_final.predict(
    X_test[caracteristicas_seleccionadas]
)

accuracy_final_test = accuracy_score(y_test, pred_final)

print("\n" + "=" * 70)
print("COMPARACIÓN FINAL")
print("=" * 70)
print(f"BASE     -> 30 features | Accuracy test = {baseline_test:.4f}")
print(
    f"CON AG   -> {len(caracteristicas_seleccionadas):02d} features | "
    f"Accuracy test = {accuracy_final_test:.4f}"
)

print("\nMatriz de confusión del modelo con características seleccionadas:")
print(confusion_matrix(y_test, pred_final))

print("\nReporte de clasificación:")
print(
    classification_report(
        y_test,
        pred_final,
        target_names=data.target_names,
        digits=4
    )
)

# =========================================================
# 12. GRÁFICAS
# =========================================================
historial_df = pd.DataFrame(historial)

plt.figure(figsize=(9, 5))
plt.plot(
    historial_df["generacion"],
    historial_df["mejor_fitness"],
    marker="o",
    label="Mejor fitness"
)
plt.plot(
    historial_df["generacion"],
    historial_df["fitness_promedio"],
    marker="o",
    label="Fitness promedio"
)
plt.xlabel("Generación")
plt.ylabel("Fitness")
plt.title("Evolución del algoritmo genético")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
plt.plot(
    historial_df["generacion"],
    historial_df["caracteristicas"],
    marker="o"
)
plt.xlabel("Generación")
plt.ylabel("Número de características")
plt.title("Cantidad de características del mejor individuo por generación")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

comparacion_accuracy = pd.DataFrame({
    "Modelo": ["Base (30 features)", "AG Feature Selection"],
    "Accuracy test": [baseline_test, accuracy_final_test]
})

plt.figure(figsize=(8, 5))
plt.bar(
    comparacion_accuracy["Modelo"],
    comparacion_accuracy["Accuracy test"]
)
plt.ylim(0.90, 1.00)
plt.ylabel("Accuracy")
plt.title("Comparación de accuracy en el conjunto de prueba")
plt.tight_layout()
plt.show()

comparacion_features = pd.DataFrame({
    "Modelo": ["Base", "AG Feature Selection"],
    "Número de features": [N_FEATURES, len(caracteristicas_seleccionadas)]
})

plt.figure(figsize=(8, 5))
plt.bar(
    comparacion_features["Modelo"],
    comparacion_features["Número de features"]
)
plt.ylabel("Número de características")
plt.title("Reducción de características")
plt.tight_layout()
plt.show()
