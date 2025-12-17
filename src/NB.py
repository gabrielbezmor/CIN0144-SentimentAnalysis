import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import shap
from preprocessing.clean_text import generate_preprocessed_dataframes
import os
import graphicFunctions



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
def train_naive_bayes(X_train, y_train, param_grid,):
    # O lime precisa dessa estrutura para funcionar automaticamente
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000)),
        ('clf', MultinomialNB())
    ])

    print("Iniciando GridSearch...")
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


# =======================================================================================
# FASE 1: COMPARAÇÃO DE PRÉ-PROCESSAMENTO (SEM TUNAGEM)
# Objetivo: Gerar dados para comparação de pré-processamentos e escolher o melhor dataset
# ========================================================================================
results_phase1 = {}
best_df_name = None
best_acc = -1

fast_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('clf', MultinomialNB())
])
df_raw = pd.read_csv('/content/dataset_raw.csv')
df_nohtml = pd.read_csv('/content/dataset_no_html.csv')
df_normalized = pd.read_csv('/content/dataset_normalized.csv')
df_corrected = pd.read_csv('/content/dataset_corrected.csv')
df_lemmatized = pd.read_csv('/content/dataset_lemmatized.csv')

dataframes = {
    "Raw": df_raw,
    "No HTML": df_nohtml,
    "Normalizado": df_normalized,
    "Correção de Erros": df_corrected,
    "Lematização": df_lemmatized
}

for name, df in dataframes.items():
    print(f"Testando dataset: {name}...", end=" ")

    X_train, X_test, y_train, y_test = prepare_data_splits(df)
    fast_pipeline.fit(X_train, y_train)

    y_pred = fast_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Guardar métricas básicas
    results_phase1[name] = acc

    print(f"Acc: {acc:.4f}")

    # Rastrear o vencedor
    if acc > best_acc:
        best_acc = acc
        best_df_name = name

print(f"\nMelhor Dataset identificado: '{best_df_name}' (Acc: {best_acc:.4f})")
print("-" * 50)

# ================================================================================
# FASE 2: TUNAGEM DE HIPERPARÂMETROS (APENAS NO VENCEDOR) E AVALIAÇÃO DE MÉTRICAS
# Objetivo: Encontrar melhores hiperparametros para modelo com maior acurácia
# ================================================================================
df_final = dataframes[best_df_name]
X_train, X_test, y_train, y_test = prepare_data_splits(df_final)

pipeline_final = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('clf', MultinomialNB())
])

param_grid = {
        'tfidf__ngram_range': [(1,1), (1, 2)], 
        'tfidf__max_df': [0.9, 1.0],            
        'tfidf__min_df': [2, 5],                 
        'clf__alpha': [0.1, 1.0]                
    }

final_model = train_naive_bayes(X_train,y_train, param_grid)
y_pred = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:,1]

results = {
    best_df_name: {
        "model": final_model,
        "accuracy": accuracy_score(y_test, y_pred),
        "X_test": X_test,
        "y_test": y_test,
    }
}

# Adicionar os outros DFs no results apenas para o gráfico de comparação (versões não tunadas)
for name, acc in results_phase1.items():
    if name != best_df_name:
        # Nota: Para os não-vencedores, não salvaremos o modelo pesado, só a acc para o gráfico
        results[name] = {"accuracy": acc}

print("\n" + "="*50)
print("RELATÓRIO FINAL")
print("="*50)
print(f"Acurácia (Teste): {results[best_df_name]['accuracy']:.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


#Plotar gráfico da acurácia x Nível de preprocessamento
print("Gerando gráfico comparativo de pré-processamento...")
graphicFunctions.plot_benchmark_results(results)

print(f"Gerando Top Keywords para o campeão: {best_df_name}")


print(f"Gerando explicação LIME para amostras do: {best_df_name}")
graphicFunctions.explain_instance_lime(results, best_df_name, index_test_set=10, num_features=15)
graphicFunctions.explain_instance_lime(results, best_df_name, index_test_set=50, num_features=15)
graphicFunctions.explain_instance_lime(results, best_df_name, index_test_set=100, num_features=15)
graphicFunctions.explain_instance_lime(results, best_df_name, index_test_set=200, num_features=15)