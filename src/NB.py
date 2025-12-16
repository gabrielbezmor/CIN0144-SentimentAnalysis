import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, train_test_split, learning_curve
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


def plot_learning_curve(estimator, title, X, y, cv=5, n_jobs=-1):
    """
    Gera o gráfico de Curva de Aprendizado.
    Isso responde à sua pergunta de 'ver a acurácia durante o treino'.
    """
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Número de exemplos de treino")
    plt.ylabel("Acurácia (Score)")

    # Gera os scores para diferentes tamanhos de dataset (10%, 30%, ..., 100%)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, 
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='accuracy'
    )

    # Calcula médias e desvio padrão
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    # Preenche a área de incerteza (std)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    # Plota as linhas
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score de Treino (Memória)")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Score de Validação Cruzada (Generalização)")

    plt.legend(loc="best")
    plt.show()


def train_naive_bayes(X_train, y_train, param_grid):

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ])

    print("Iniciando GridSearch (Naïve Bayes)...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
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