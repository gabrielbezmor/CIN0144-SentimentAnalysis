import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from lime.lime_text import LimeTextExplainer
import shap


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