import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from preprocessing.clean_dataset import remove_duplicates
from preprocessing.clean_text import remove_stopwords, tokenize, load_nlp
from preprocessing.feature_extraction import add_features
from sklearn.feature_extraction.text import CountVectorizer
"""
EDA - Análise Exploratória de Dados
Inspiração: https://medium.com/dscier/eda-nlp-fe483c6871ba
"""
def load_data(file_path):
    """
    Carrega o dataset de um arquivo CSV.
    Retorna um dataframe pandas.
    """
    dataframe = pd.read_csv(file_path)
    return dataframe

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
            i,                      # posição x
            value + (value * 0.01), # posição y (um pouco acima da barra)
            str(value),             # texto
            ha='center', va='bottom', fontsize=10
        )
    plt.show()


def generate_wordcloud(dataframe, text_column):
    """
    Gera e exibe uma nuvem de palavras a partir de uma coluna de texto
    no dataframe.
    Bom para visualização rápida do conteúdo textual.
    """
    text = ' '.join(dataframe[text_column].dropna().astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Nuvem de Palavras para {text_column}')
    plt.show()

def create_duplicate_csv(dataframe):
    """
    Cria um csv com as linhas duplicadas.
    Mostra as cópias individuais de cada linha duplicada.
    """
    duplicates = df_train[df_train.duplicated(keep=False)]
    with pd.option_context('display.max_rows', None):
        duplicates.to_csv('data/raw/duplicates.csv', index=False)

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

def plot_text_length_distribution(dataframe, text_column='text', bins=50, figsize=(10, 6), show_percentiles=(25, 50, 75)):
    """
    Plota a distribuição dos comprimentos (número de caracteres) dos textos
    - bins: número de bins para o histograma
    - figsize: tamanho da figura
    - show_percentiles: tupla com percentis a exibir na legenda
    """

    lengths = dataframe[text_column].dropna().astype(str).str.len()
    if lengths.empty:
        print("Nenhuma entrada encontrada para calcular comprimentos.")
        return

    # estatísticas rápidas
    mean = lengths.mean()
    median = lengths.median()
    std = lengths.std()
    pcts = {p: int(lengths.quantile(p/100.0)) for p in show_percentiles}

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize, gridspec_kw={"height_ratios": (4, 1)}, tight_layout=True)

    axes[0].hist(lengths, bins=bins, color='tab:blue', alpha=0.75)
    axes[0].set_title(f'Distribuição dos comprimentos de "{text_column}"')
    axes[0].set_xlabel("Comprimento (número de caracteres)")
    axes[0].set_ylabel("Frequência")

    # desenha linhas de estatísticas
    axes[0].axvline(mean, color='orange', linestyle='--', label=f'mean={mean:.1f}')
    axes[0].axvline(median, color='green', linestyle='--', label=f'median={median:.0f}')
    for p, val in pcts.items():
        axes[0].axvline(val, linestyle=':', label=f'{p}th={val}')

    axes[0].legend()

    # boxplot compacto embaixo para outliers / dispersão
    axes[1].boxplot(lengths, vert=False)
    axes[1].set_xlabel("Comprimento (número de caracteres)")
    axes[1].set_yticks([])

    plt.show()

    # imprime resumo numérico
    stats_msg = f"count={len(lengths)}, mean={mean:.1f}, median={median:.0f}, std={std:.1f}, min={lengths.min()}, max={lengths.max()}"
    pct_msg = "  ".join([f"{p}th={v}" for p, v in pcts.items()])
    print(stats_msg)
    print(pct_msg)


def plot_word_count_distribution(dataframe, text_column='text', bins=50, figsize=(10, 6), show_percentiles=(25, 50, 75)):
    """
    Plota a distribuição dos comprimentos (número de palavras) das entradas
    em `text_column` do dataframe.
    - bins: número de bins para o histograma
    - figsize: tamanho da figura
    """
    counts = dataframe[text_column].dropna().astype(str).str.split().apply(len)
    if counts.empty:
        print("Nenhuma entrada encontrada para calcular comprimentos.")
        return

    # estatísticas rápidas
    mean = counts.mean()
    median = counts.median()
    std = counts.std()
    pcts = {p: int(counts.quantile(p/100.0)) for p in show_percentiles}

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize, gridspec_kw={"height_ratios": (4, 1)}, tight_layout=True)

    axes[0].hist(counts, bins=bins, color='tab:blue', alpha=0.75)
    axes[0].set_title(f'Distribuição dos comprimentos de "{text_column}"')
    axes[0].set_xlabel("Comprimento (número de palavras)")
    axes[0].set_ylabel("Frequência")

    # desenha linhas de estatísticas
    axes[0].axvline(mean, color='orange', linestyle='--', label=f'mean={mean:.1f}')
    axes[0].axvline(median, color='green', linestyle='--', label=f'median={median:.0f}')
    for p, val in pcts.items():
        axes[0].axvline(val, linestyle=':', label=f'{p}th={val}')

    axes[0].legend()

    # boxplot compacto embaixo para outliers / dispersão
    axes[1].boxplot(counts, vert=False)
    axes[1].set_xlabel("Comprimento (número de palavras)")
    axes[1].set_yticks([])

    plt.show()

    # imprime resumo numérico
    stats_msg = f"count={len(counts)}, mean={mean:.1f}, median={median:.0f}, std={std:.1f}, min={counts.min()}, max={counts.max()}"
    pct_msg = "  ".join([f"{p}th={v}" for p, v in pcts.items()])
    print(stats_msg)
    print(pct_msg)

def print_eda_summary(df_list=[], names=[], msg=""):
    for df, name in zip(df_list, names):
        print(f"{name} set: " + msg)
        summary = eda_summary(df)
        for key, value in summary.items():
            print(f"{key}: {value}\n")

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

if __name__ == "__main__":
    df_train = pd.read_csv('data/raw/Train.csv')
    
    df_val = pd.read_csv('data/raw/Valid.csv')
    
    df_test = pd.read_csv('data/raw/Test.csv')
    
    df_full = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)

    print_eda_summary([df_full], ['Completo'])

    check_duplicates_between_sets([df_train, df_val, df_test], ['Treino', 'Validação', 'Teste'])

    df_clean = remove_duplicates(df_full)
    print_eda_summary([df_clean], ['Limpo'], msg="após remoção de duplicatas")
    df_clean = add_features(df_clean, input_column='text')
    
    show_distribution(df_clean, 'label')

    
    nlp = load_nlp()
    tokenized = df_train['text'].dropna().astype(str).apply(lambda x: tokenize(x, nlp=nlp))
    remove_stopwords_list = tokenized.apply(lambda x: ' '.join(remove_stopwords(x)))

    plot_top_ngrams(remove_stopwords_list, ngram_range=(1,1), title="Top 10 Unigrams")
    plot_top_ngrams(remove_stopwords_list, ngram_range=(2,2), title="Top 10 Bigrams")
    plot_top_ngrams(remove_stopwords_list, ngram_range=(3,3), title="Top 10 Trigrams")