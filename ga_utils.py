import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def cargar_datos_base(seed=42, test_size=0.20):
    """
    Carga el dataset Breast Cancer Wisconsin (Diagnostic) y realiza 
    una división estratificada Train/Test (80/20) para evitar data leakage.
    
    Retorna:
        X_train, X_test, y_train, y_test, feature_names, target_names
    """
    # 1. Cargar dataset oficial
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target
    
    # 2. Split estratificado (80% Train, 20% Test) con semilla fija
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=seed, 
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test, list(data.feature_names), list(data.target_names)

if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, feats, targets = cargar_datos_base()
    print("✓ ga_utils.py ejecutado correctamente.")