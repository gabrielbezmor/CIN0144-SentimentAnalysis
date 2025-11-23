import re
import spacy
from transformers import BertTokenizer

"""
Contém funções úteis para pré-processamento de texto.
A ideia é treinar os modelos com subconjuntos dessas funções aplicadas
sobre os dados e analisar os efeitos no desempenho.
O melhor modelo não necessariamente usará todas as técnicas.
Lembrar SEMPRE de aplicar as mesmas técnicas nos dados de validação e teste.
Nunca sobrescrever dados, sempre criar novos dataframes/colunas.
"""
def bert_tokenize(text):
    """
    Tokeniza o texto usando o tokenizador BERT
    (não sei se é necessário ou se o transformer já faz isso sozinho)
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer.tokenize(text)
    return tokens

def tokenize(text):
    """
    Tokeniza o texto em palavras individuais (para modelos tradicionais)
    """
    tokens = spacy.word_tokenize(text)
    return tokens

def normalize_whitespace(text):
    """
    Substitui múltiplos espaços em branco por um único espaço e remove espaços extras no início/fim
    """
    return re.sub(r'\s+', ' ', text).strip()

def remove_special_characters(text):
    """
    Remove caracteres especiais, mantendo letras, números e espaços
    """
    return re.sub(r'[^A-Za-z0-9\s]', '', text)

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
def lemmatize_text(text, nlp=nlp):    
    """
    Reduz palavras à sua forma base
    """
    doc = nlp(text)
    lemmatized = ' '.join([token.lemma_ for token in doc])
    return lemmatized

def remove_stopwords(tokenized_text):
    """
    Mantém apenas palavras importantes
    (a implementar)
    """

def handle_typos(text):
    """
    Corrige erros de digitação comuns
    (a implementar) 
    (talvez sejam necessárias várias funções com essa
    finalidade, mas diferentes níveis de "agressividade" na correção)
    """