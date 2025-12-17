# Gráfico Acuáracia x Nivel de processamento
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from lime.lime_text import LimeTextExplainer


def plot_benchmark_results(results_dict):
    # 1. Converter dicionário para DataFrame para facilitar o plot
    data = []
    for name, res in results_dict.items():
        data.append({
            'Dataset Version': name,
            'Accuracy': res['accuracy']
        })

    df_res = pd.DataFrame(data).sort_values(by='Accuracy', ascending=False)

    # 2. Configurar o estilo
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # 3. Criar o gráfico de barras
    ax = sns.barplot(data=df_res, x='Dataset Version', y='Accuracy', palette='viridis')

    # 4. Adicionar os valores em cima das barras
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.4f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points',
                    fontweight='bold')

    plt.title('Impacto do Pré-processamento na Acurácia do Modelo', fontsize=14)
    plt.ylim(0, 1.1)  # Deixa um respiro em cima
    plt.ylabel('Acurácia')
    plt.xticks(rotation=45)  # Rotaciona labels se forem longos
    plt.tight_layout()
    plt.show()

# Explicabilidade com LIME
def explain_instance_lime(results_dict, model_name, index_test_set, num_features=6):

    # Recuperar os artefatos
    data = results_dict[model_name]
    pipeline = data['model']
    X_test_slice = data['X_test']  # Série pandas com texto cru
    y_test_slice = data['y_test']

    # Pegar o texto específico e a label real
    text_instance = X_test_slice.iloc[index_test_set]
    true_label = y_test_slice.iloc[index_test_set]

    # Instanciar o Explainer
    explainer = LimeTextExplainer(class_names=['Negativo', 'Positivo'])  # Ajuste nomes das classes se precisar

    # Gerar explicação
    # O pipeline.predict_proba cuida da vetorização internamente
    exp = explainer.explain_instance(
        text_instance,
        pipeline.predict_proba,
        num_features=num_features
    )

    print(f"--- Análise LIME para modelo: {model_name} ---")
    print(f"Texto: {text_instance}")
    print(f"Label Real: {true_label}")
    print(f"Predição do Modelo: {pipeline.predict([text_instance])[0]}")

    exp.as_pyplot_figure()
    plt.title(f"LIME: Importância das palavras ({model_name})")
    plt.show()


def show_top_keywords(results_dict, model_name, n=10):
    model_data = results_dict[model_name]
    pipeline = model_data['model']

    # Extrair o vetorizador e o classificador do pipeline
    vectorizer = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']

    # Pegar nomes das features (palavras) e coeficientes
    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_.flatten()

    # Criar dataframe para ordenar
    df_coefs = pd.DataFrame({'word': feature_names, 'coef': coefs})

    # Top Positivas e Negativas
    top_positive = df_coefs.sort_values(by='coef', ascending=False).head(n)
    top_negative = df_coefs.sort_values(by='coef', ascending=True).head(n)

    # Plotar
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Cores: Azul para positivo, Vermelho para negativo
    sns.barplot(data=top_positive, x='coef', y='word', ax=axes[0], color='blue')
    axes[0].set_title(f'Top {n} Palavras que puxam para Classe 1 (Positivo)')

    sns.barplot(data=top_negative, x='coef', y='word', ax=axes[1], color='red')
    axes[1].set_title(f'Top {n} Palavras que puxam para Classe 0 (Negativo)')

    plt.tight_layout()
    plt.show()


#FUNÇÕES DO NAIVE BAYES
def mostrar_palavras_reais_do_sentimento(vectorizer, clf, n=15):

    feature_names = vectorizer.get_feature_names_out()


    neg_prob = clf.feature_log_prob_[0] 
    pos_prob = clf.feature_log_prob_[1] 


    diferenca = pos_prob - neg_prob


    top_pos_indices = np.argsort(diferenca)[-n:][::-1]

    top_neg_indices = np.argsort(diferenca)[:n]

    print(f"--- Top {n} Palavras que indicam POSITIVO (O que faz o modelo amar) ---")
    for i in top_pos_indices:
        print(f"{feature_names[i]}: {diferenca[i]:.4f}")

    print(f"\n--- Top {n} Palavras que indicam NEGATIVO (O que faz o modelo odiar) ---")
    for i in top_neg_indices:
        print(f"{feature_names[i]}: {diferenca[i]:.4f}")

def plot_palavras_importantes(vectorizer, clf, n=20):
    """
    Plota um gráfico de barras divergentes com as palavras mais importantes
    para Positivo e Negativo no Naive Bayes.
    """

    feature_names = vectorizer.get_feature_names_out()
    neg_prob = clf.feature_log_prob_[0]
    pos_prob = clf.feature_log_prob_[1]


    diferenca = pos_prob - neg_prob


    df_features = pd.DataFrame({
        'feature': feature_names,
        'importance': diferenca
    })


    df_pos = df_features.nlargest(n, 'importance')
    df_neg = df_features.nsmallest(n, 'importance')


    df_plot = pd.concat([df_neg.sort_values('importance', ascending=False), 
                            df_pos.sort_values('importance', ascending=True)])


    plt.figure(figsize=(12, 10))


    colors = ['red' if x < 0 else 'green' for x in df_plot['importance']]

    plt.barh(df_plot['feature'], df_plot['importance'], color=colors, alpha=0.7)


    plt.title(f'Top {n} Palavras mais Determinantes por Sentimento (Naive Bayes)', fontsize=15)
    plt.xlabel('Importância (Log-Odds Ratio)', fontsize=12)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8) 
    plt.grid(axis='x', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()
