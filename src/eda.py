import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from preprocessing.clean_dataset import remove_duplicates
from preprocessing.clean_text import remove_stopwords, tokenize, load_nlp, normalize_whitespace, remove_special_characters, filter_text_with_emoji
from preprocessing.feature_extraction import add_features
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import outlier_detection as od
"""
EDA - Análise Exploratória de Dados
Inspiração: https://medium.com/dscier/eda-nlp-fe483c6871ba
"""
def eda_summary(dataframe):
    """
    Mostra resumo das características principais do dataframe.
    shape, colunas, tipos de dados e valores ausentes.
    """
    summary = {}
    summary['shape'] = dataframe.shape
    summary['columns'] = dataframe.columns.tolist()
    summary['dtypes'] = dataframe.dtypes.to_dict()
    summary['missing_values'] = dataframe.isnull().sum().to_dict()
    summary['duplicate_rows'] = dataframe.duplicated().sum()
    return summary


def generate_wordcloud(dataframe, text_column):
    """
    Gera e exibe uma nuvem de palavras a partir de uma coluna de texto
    no dataframe.
    Bom para visualização rápida do conteúdo textual.
    """
    text = ' '.join(dataframe[text_column].dropna().astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=None).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def plot_top_ngrams(corpus, n_top=10, ngram_range=(1,1), title=f"Top n-grams"):
    """
    Plota os n-gramas mais frequentes.
    """
    vec = CountVectorizer(ngram_range=ngram_range, stop_words=None).fit(corpus)
    bag = vec.transform(corpus)
    counts = bag.sum(axis=0).A1
    vocab = vec.get_feature_names_out()
    freq = pd.DataFrame({"ngram": vocab, "count": counts})
    freq = freq.sort_values("count", ascending=False).head(n_top)

    plt.figure(figsize=(10,6))
    plt.barh(freq["ngram"], freq["count"])
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()

def show_distribution(dataframe, target_column):
    """
    Plota a distribuição dos valores da coluna alvo do dataframe.
    """
    plt.figure(figsize=(8, 6))
    dataframe[target_column].value_counts().plot(kind='bar')
    counts = dataframe[target_column].value_counts().sort_index()
    ax = counts.plot(kind='bar')
    plt.title(f'Distribuicao de {target_column}')
    plt.xlabel(target_column)
    plt.ylabel('Frequencia')
    for i, value in enumerate(counts):
        ax.text(
            i,                      
            value + (value * 0.01),
            str(value),            
            ha='center', va='bottom', fontsize=10
        )
    plt.show()

def plot_text_length_distribution(target_column, msg, bins=50, figsize=(10, 6), show_percentiles=(25, 50, 75)):
    """
    Plota a distribuição dos comprimentos (número de caracteres) dos textos
    - bins: número de bins para o histograma
    - figsize: tamanho da figura
    - show_percentiles: tupla com percentis a exibir na legenda
    """

    # estatísticas rápidas
    mean = target_column.mean()
    median = target_column.median()
    std = target_column.std()
    pcts = {p: int(target_column.quantile(p/100.0)) for p in show_percentiles}

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize, gridspec_kw={"height_ratios": (4, 1)}, tight_layout=True)

    axes[0].hist(target_column, bins=bins, color='tab:blue', alpha=0.75)
    axes[0].set_title(f'Distribuição de {msg}')
    axes[0].set_xlabel(msg)
    axes[0].set_ylabel("Frequência")

    # desenha linhas de estatísticas
    axes[0].axvline(mean, color='orange', linestyle='--', label=f'mean={mean:.1f}')
    axes[0].axvline(median, color='green', linestyle='--', label=f'median={median:.0f}')
    for p, val in pcts.items():
        axes[0].axvline(val, linestyle=':', label=f'{p}th={val}')

    axes[0].legend()

    # boxplot compacto embaixo para outliers / dispersão
    axes[1].boxplot(target_column, vert=False)
    axes[1].set_xlabel(msg)
    axes[1].set_yticks([])

    plt.show()

    # imprime resumo numérico
    stats_msg = f"count={len(target_column)}, mean={mean:.1f}, median={median:.0f}, std={std:.1f}, min={target_column.min()}, max={target_column.max()}"
    pct_msg = "  ".join([f"{p}th={v}" for p, v in pcts.items()])
    print(stats_msg)
    print(pct_msg)

def check_duplicates_between_sets(dataframes=[], names=[]):
    """
    Verifica e imprime o número de linhas duplicadas entre dataframes.
    """
   #remove duplicatas internas
    no_internal_duplicates = []
    for df in dataframes:
        df_clean = remove_duplicates(df) 
        no_internal_duplicates.append(df_clean)
        
    #cria datasets combinados sem duplicatas internas
    combinations = []
    comb_names = []
    for i, (df1, name1) in enumerate(zip(no_internal_duplicates, names)):
        for j, (df2, name2) in enumerate(zip(no_internal_duplicates, names)):
            if i < j:
                combined = pd.concat([df1, df2]).reset_index(drop=True)
                name_comb = name1 + ' e ' + name2
                combinations.append(combined)
                comb_names.append(name_comb)

    for df, name in zip(combinations, comb_names):
        duplicated = df[df.duplicated(keep=False)]
        print(f"Linhas duplicadas entre {name} {len(duplicated)}")


def find_and_save_outliers(dataframe, numeric_columns, method='iqr', output_dir='data/analysis'):
    """
    Detecta outliers em múltiplas colunas numéricas e salva relatório.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    all_outliers = {}
    for col in numeric_columns:
        if col in dataframe.columns:
            outliers = od.get_outliers_by_method(dataframe[col], method=method)
            all_outliers[col] = outliers
            print(f"{col}: {len(outliers)} outliers detectados (método: {method})")
    
    # salvar índices de outliers de cada coluna
    outlier_indices = set()
    for col, outliers in all_outliers.items():
        outlier_indices.update(outliers.index)
    
    outlier_rows = dataframe.loc[list(outlier_indices), :].copy()
    outlier_rows.to_csv(f'{output_dir}/{method}_outliers.csv', index=False)
    print(f"\nOutliers salvos em: {output_dir}/{method}_outliers.csv ({len(outlier_rows)} linhas)")
    
    return outlier_rows, all_outliers
if __name__ == "__main__":
    RANDOM_SEED = 42
    # CARREGAR DATASETS
    df_train = pd.read_csv('data/raw/Train.csv')
    df_val = pd.read_csv('data/raw/Valid.csv')
    df_test = pd.read_csv('data/raw/Test.csv')
    
    # ANÁLISE ESTRUTURAL DO DATASET
    #check_duplicates_between_sets([df_train, df_val, df_test], ['Treino', 'Validação', 'Teste'])
    df_full = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)
    
    eda_summary(df_full)

    #generate_wordcloud(df_full, 'text')
    #plot_top_ngrams(df_full['text'].astype(str), 20, (1,1), "")

    df_clean = remove_duplicates(df_full)
    #eda_summary(df_clean)
    df_clean = add_features(df_clean, input_column='text')
    
    
    #show_distribution(df_clean, 'label')
    #plot_text_length_distribution(df_clean['length'], msg='Número de caracteres')
    #plot_text_length_distribution(df_clean['word_count'], msg='Contagem de Palavras')
    #plot_text_length_distribution(df_clean['mean_word_length'], msg='Comprimento Médio das Palavras')

    print("Assimetria comprimento: {}".format(df_clean['length'].skew())) 
    print("Curtose comprimento: {}".format(df_clean['length'].kurtosis()))

    print("Assimetria contagem de palavras: {}".format(df_clean['word_count'].skew())) 
    print("Curtose contagemm de palavras: {}".format(df_clean['word_count'].kurtosis()))

    print("Assimetria comprimento médio de palavras: {}".format(df_clean['mean_word_length'].skew())) 
    print("Curtose comprimento médio de palavras: {}".format(df_clean['mean_word_length'].kurtosis()))

    

    #print(df_clean.nlargest(5,['mean_word_length']))
    #print(df_clean.nsmallest(5, ['mean_word_length']))

    #print(df_clean.nlargest(5,['length']))
    #print(df_clean.nsmallest(5, ['length']))

    #print(df_clean.nlargest(5,['word_count']))
    #print(df_clean.nsmallest(5, ['word_count']))
    # PRÉ-PROCESSAMENTO BÁSICO
    
    
    #emojis = filter_text_with_emoji(df_clean['text'].astype(str))
    #print(f"Encontradas {len(emojis)} linhas com emojis.")
    
    #Análise univariada
    

    #Separação dos conjuntos para evitar vazamento de dados
    #X_train, X_test, y_train, y_test = train_test_split(df_clean['text'], df_clean['label'], test_size=0.2, random_state=RANDOM_SEED, stratify=df_clean['label'])
    #X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=RANDOM_SEED, stratify=y_test)
    nlp = load_nlp()
    corpus = df_clean['text'].astype(str)
    


    
    lowercase = corpus.str.lower()
    normalized = lowercase.apply(lambda x: normalize_whitespace(x))
    tokenized = normalized.apply(lambda x: tokenize(x, nlp=nlp))
    no_stopwords = tokenized.apply(lambda x: remove_stopwords(x))


    # DETECÇÃO DE OUTLIERS
   
    # Comparar métodos
    print("\n=== Comparando métodos para 'length' ===")
    comparison = od.compare_outlier_methods(df_clean['length'])
    consensus_count = comparison['consensus'].sum()
    print(f"Pontos acusados por 2+ métodos: {consensus_count}")
    outlier_rows = df_clean[comparison['consensus']]
    outlier_rows.to_csv('data/analysis/outliers_length.csv', index=False)
    
    print("\n=== Comparando métodos para 'word_count' ===")
    comparison = od.compare_outlier_methods(df_clean['word_count'])
    consensus_count = comparison['consensus'].sum()
    print(f"Pontos acusados por 2+ métodos: {consensus_count}")
    outlier_rows = df_clean[comparison['consensus']]
    outlier_rows.to_csv('data/analysis/outliers_word_count.csv', index=False)

    

    nlp = load_nlp()

    plot_top_ngrams(no_stopwords, ngram_range=(1,1), title="Top 10 Unigrams")
    plot_top_ngrams(no_stopwords, ngram_range=(2,2), title="Top 10 Bigrams")
    plot_top_ngrams(no_stopwords, ngram_range=(3,3), title="Top 10 Trigrams")