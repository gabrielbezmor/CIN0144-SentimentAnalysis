import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from lime.lime_text import LimeTextExplainer
from preprocessing.clean_dataset import combine_data, remove_duplicates
import shap


# Supondo dados já normalizados
def train_SVM(X_train, y_train, param_grid):
    # O lime precisa dessa estrutura para funcionar automaticamente
    pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', SVC(probability=True, random_state=42))
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

def train():

    param_grid = {
    "clf__kernel": ["linear"],
    "clf__C": [0.1, 1, 10, 100, 1000]
}

    df = combine_data('/CIN0144-SentimentAnalysis/data/raw/Train.csv', '/CIN0144-SentimentAnalysis/data/raw/Valid.csv', '/CIN0144-SentimentAnalysis/data/raw/Test.csv')
    df = remove_duplicates(df)
    x = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    
    train_SVM(X_train, y_train, param_grid)


