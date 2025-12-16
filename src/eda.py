import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from preprocessing.clean_dataset import remove_duplicates
from preprocessing.clean_text import remove_stopwords, tokenize, load_nlp, normalize_whitespace, remove_html, clean_text_pipeline, customize_stopwords
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing.feature_extraction import add_features
import outlier_detection as od
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import plotly.express as px
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

def generate_wordcloud(text):
    """
    Gera wordcloud a partir de string
    """
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=None).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def generate_wordcloud_df(dataframe, text_column):
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

def plot_text_length_distribution(df, col, msg, bins=50, figsize=(10, 6), show_percentiles=(25, 50, 75)):
    """
    Plota a distribuição dos comprimentos (número de caracteres) dos textos
    - bins: número de bins para o histograma
    - figsize: tamanho da figura
    - show_percentiles: tupla com percentis a exibir na legenda
    """

    # estatísticas rápidas
    mean = df[col].mean()
    median = df[col].median()
    std = df[col].std()
    pcts = {p: int(df[col].quantile(p/100.0)) for p in show_percentiles}

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize, gridspec_kw={"height_ratios": (4, 1)}, tight_layout=True)

    axes[0].hist(df[col], bins=bins, color='tab:blue', alpha=0.75)
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
    axes[1].boxplot(df[col], vert=False)
    axes[1].set_xlabel(msg)
    axes[1].set_yticks([])

    plt.show()

    # imprime resumo numérico
    stats_msg = f"count={len(df[col])}, mean={mean:.1f}, median={median:.0f}, std={std:.1f}, min={df[col].min()}, max={df[col].max()}"
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

def scatterplot(df, col1, col2):

    plt.scatter(df[col1], df[col2])
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title(f"Scatter plot de {col1} vs {col2}")
    plt.show()

def estatisticas_univariadas(df, col):
    print(f"Assimetria {col}: {df[col].skew()}")
    print(f"Curtose {col}: {df[col].kurtosis()}")


def corr_matrix(df, method='pearson'):
    # Apenas colunas numéricas
    corr = df.select_dtypes(include='number').corr(method=method)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Colormap mais brando
    cax = ax.imshow(
        corr,
        cmap='coolwarm',
        vmin=-1,
        vmax=1
    )

    # Barra de cores
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    # Ticks e labels
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))

    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)

    # Título
    ax.set_title("Matriz de Correlação", pad=20)

    # Anotar valores em cada célula
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(
                j, i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9
            )

    plt.tight_layout()
    plt.show()

def semantic_similarity(df, col):
    sentences = df[col]

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embedding_imdb = model.encode(sentences)

    # generate TSNE - 2D
    X = list(embedding_imdb["text_lower_embedding"])
    X_embedded = TSNE(n_components=2).fit_transform(X)
    df_proj_embeddings = pd.DataFrame(X_embedded)
    df_proj_embeddings = df_proj_embeddings.rename(columns={0:'x',1:'y'})
    df_proj_embeddings['label'] = embedding_imdb["Category"]

    # plot 
    fig = px.scatter(df_proj_embeddings, x="x", y="y", color="label", width=600, height=400)
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="White",
        plot_bgcolor="White",
    )
    fig.show()

    # generate TSNE - 3D
    tsne_3D = TSNE(n_components=3, random_state=0)
    projections_3D = tsne_3D.fit_transform(X)
    df_3D_proj_embeddings = pd.DataFrame(projections_3D)
    df_3D_proj_embeddings = df_3D_proj_embeddings.rename(columns={0:'x',1:'y', 2:'z'})
    df_3D_proj_embeddings['label'] = embedding_imdb["Category"]

    # plot
    fig = px.scatter_3d(
        df_3D_proj_embeddings, x="x", y="y", z="z", color="label", width=600, height=400
    )
    fig.update_traces(marker_size=8)
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="White",
        plot_bgcolor="White",
    )
    fig.show()

if __name__ == "__main__":
    RANDOM_SEED = 42
# CARREGAR DATASETS
    df_train = pd.read_csv('data/raw/Train.csv')
    df_val = pd.read_csv('data/raw/Valid.csv')
    df_test = pd.read_csv('data/raw/Test.csv')
    

    # ANÁLISE ESTRUTURAL DO DATASET
    #check_duplicates_between_sets([df_train, df_val, df_test], ['Treino', 'Validação', 'Teste'])
    df_full = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)
    
    #eda_summary(df_full)

    #generate_wordcloud_df(df_full, 'text')
    #plot_top_ngrams(df_full['text'].astype(str), 20, (1,1), "")
    
    df_clean = remove_duplicates(df_full)

    
    #eda_summary(df_clean)
   

    #print(df_clean.nlargest(5,['mean_word_length']))
    #print(df_clean.nsmallest(5, ['mean_word_length']))

    #print(df_clean.nlargest(5,['length']))
    #print(df_clean.nsmallest(5, ['length']))

    #print(df_clean.nlargest(5,['word_count']))
    #print(df_clean.nsmallest(5, ['word_count']))
    
    
    #emojis = filter_text_with_emoji(df_clean['text'].astype(str))
    #print(f"Encontradas {len(emojis)} linhas com emojis.")
    
    
    #ANÁLISE UNIVARIADA
    
    df_clean = add_features(df_clean, input_column='text')
    
    #show_distribution(df_clean, 'label')

    #plot_text_length_distribution(df_clean, 'length', msg='Número de caracteres')
    #plot_text_length_distribution(df_clean,'word_count', msg='Contagem de Palavras')
    #plot_text_length_distribution(df_clean,'mean_word_length', msg='Comprimento Médio das Palavras')
    #estatisticas_univariadas(df_clean, 'length')
    #estatisticas_univariadas(df_clean, 'word_count')
    #estatisticas_univariadas(df_clean, 'mean_word_length')

    nlp = load_nlp()
    #corpus = df_clean['text'].astype(str)
    #lowercase = corpus.str.lower()
    #nohtml = lowercase.apply(lambda x : remove_html(x))
    #normalized = nohtml.apply(lambda x: normalize_whitespace(x))
    #tokenized = normalized.apply(lambda x: tokenize(x, nlp=nlp))
    #no_stopwords = tokenized.apply(lambda x: remove_stopwords(x))

    #plot_top_ngrams(no_stopwords, ngram_range=(1,1), title="Top 10 Unigrams")
    #plot_top_ngrams(no_stopwords, ngram_range=(2,2), title="Top 10 Bigrams")
    #plot_top_ngrams(no_stopwords, ngram_range=(3,3), title="Top 10 Trigrams")

    
    # DETECÇÃO DE OUTLIERS
   
    # Comparar métodos
    #print("\n=== Comparando métodos para 'length' ===")
    #comparison = od.compare_outlier_methods(df_clean['length'])
    #consensus_count = comparison['consensus'].sum()
    #print(f"Pontos acusados por 2+ métodos: {consensus_count}")
    #outlier_rows = df_clean[comparison['consensus']]
    #outlier_rows.to_csv('data/analysis/outliers_length.csv', index=False)
    
    #print("\n=== Comparando métodos para 'word_count' ===")
    #comparison = od.compare_outlier_methods(df_clean['word_count'])
    #consensus_count = comparison['consensus'].sum()
    #print(f"Pontos acusados por 2+ métodos: {consensus_count}")
    #outlier_rows = df_clean[comparison['consensus']]
    #outlier_rows.to_csv('data/analysis/outliers_word_count.csv', index=False)

    #print("\n=== Comparando métodos para 'mean_word_length' ===")
    #comparison = od.compare_outlier_methods(df_clean['mean_word_length'])
    #consensus_count = comparison['consensus'].sum()
    #print(f"Pontos acusados por 2+ métodos: {consensus_count}")
    #outlier_rows = df_clean[comparison['consensus']]
    #outlier_rows.to_csv('data/analysis/outliers_mean_word_length.csv', index=False)


    #ANÁLISE BIVARIADA

    #df_neg = df_clean[df_clean['label']==0]
    #df_pos = df_clean[df_clean['label']==1]

    #corpus_neg = df_neg['text'].astype(str)
    #lowercase_neg = corpus_neg.str.lower()
    #nohtml_neg = lowercase_neg.apply(lambda x : remove_html(x))
    #normalized_neg = nohtml_neg.apply(lambda x: normalize_whitespace(x))
    #tokenized_neg = normalized_neg.apply(lambda x: tokenize(x, nlp=nlp))
    #no_stopwords_neg = tokenized_neg.apply(lambda x: remove_stopwords(x))

    #plot_top_ngrams(no_stopwords_neg, ngram_range=(1,1), title="Top 10 Unigrams dos Negativos")
    #plot_top_ngrams(no_stopwords_neg, ngram_range=(2,2), title="Top 10 Bigrams dos Negativos")
    #plot_top_ngrams(no_stopwords_neg, ngram_range=(3,3), title="Top 10 Trigrams dos Negativos")

    #corpus_pos = df_pos['text'].astype(str)
    #lowercase_pos = corpus_pos.str.lower()
    #nohtml_pos = lowercase_pos.apply(lambda x : remove_html(x))
    #normalized_pos = nohtml_pos.apply(lambda x: normalize_whitespace(x))
    #tokenized_pos = normalized_pos.apply(lambda x: tokenize(x, nlp=nlp))
    #no_stopwords_pos = tokenized_pos.apply(lambda x: remove_stopwords(x))

    #plot_top_ngrams(no_stopwords_pos, ngram_range=(1,1), title="Top 10 Unigrams dos Positivos")
    #plot_top_ngrams(no_stopwords_pos, ngram_range=(2,2), title="Top 10 Bigrams dos Positivos")
    #plot_top_ngrams(no_stopwords_pos, ngram_range=(3,3), title="Top 10 Trigrams dos Positivos")

    #plot_text_length_distribution(df_neg,'length', msg="comprimento na classe 0")
    #plot_text_length_distribution(df_neg,'word_count', msg="número de palavras na classe 0")
    #plot_text_length_distribution(df_neg,'mean_word_length', msg="comprimento médio das palavras na classe 0")

    
    #plot_text_length_distribution(df_pos,'length', msg="comprimento na classe 1")
    #plot_text_length_distribution(df_pos,'word_count', msg="número de palavras na classe 1")
    #plot_text_length_distribution(df_pos,'mean_word_length', msg="comprimento médio das palavras na classe 1")

    #scatterplot(df_clean, 'length', 'label')
    #scatterplot(df_clean, 'word_count', 'label')
    #scatterplot(df_clean, 'mean_word_length', 'label')
    
    #corr_matrix(df_clean.drop(columns=['text']))

    #ANÁLISE MULTIVARIADA:
    nlp = customize_stopwords(nlp)
    df_clean['clean_text'] = df_clean['text'].apply(lambda x : clean_text_pipeline(nlp, x))
    semantic_similarity(df_clean, 'text')