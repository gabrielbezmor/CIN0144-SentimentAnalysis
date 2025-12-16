import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from lime.lime_text import LimeTextExplainer
import shap


def prepare_data_splits(df):
    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X,y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Dados prontos!")
    print(f"Treino (será usado no CV): {len(X_train)} amostras")
    print(f"Teste (Cofre fechado): {len(X_test)} amostras")
    return X_train, X_test, y_train, y_test

# Supondo dados já normalizados
def train_logistic_regression(X_train, y_train, param_grid):
    # O lime precisa dessa estrutura para funcionar automaticamente
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])

    print("Iniciando GridSearch...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='f1_weighted',  # F1 é melhor se houver desbalanceamento
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    print("Melhores parâmetros:", grid_search.best_params_)
    return grid_search.best_estimator_

def lime_explainer(pipeline, instance,class_names):
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(
        instance,
        pipeline.predict_proba,
        num_features=10
    )

    return exp