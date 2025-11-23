import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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
    plt.title(f'Distribuicao de {target_column}')
    plt.xlabel(target_column)
    plt.ylabel('Frequencia')
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

def get_ngram(text, n):
    """
    Gera n-gramas a partir de um texto.
    Retorna uma lista de n-gramas.
    n-gramas mostram sequências de n palavras que aparecem juntas
    (a implementar)
    """



if __name__ == "__main__":
    df_train = pd.read_csv('data/raw/Train.csv')
    
    
    
    summary = eda_summary(df_train)
    for key, value in summary.items():
        print(f"{key}: {value}\n")

    show_distribution(df_train, 'label')

    generate_wordcloud(df_train, 'text')