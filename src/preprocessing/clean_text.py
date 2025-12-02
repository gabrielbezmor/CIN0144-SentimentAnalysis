import re
import spacy
from spacy.lang.en import English
import torch
import en_core_web_sm
from transformers import BertTokenizer
from bs4 import BeautifulSoup
from symspellpy import SymSpell, Verbosity
import string
import pkg_resources

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

def load_nlp():
    """
    Carrega o modelo spaCy para inglês
    """
    nlp = en_core_web_sm.load()
    return nlp

def tokenize(text, nlp):
    """
    Tokeniza o texto em palavras individuais (para modelos tradicionais)
    """
    tokens = nlp.tokenizer(text)
    return tokens

def print_tokenized(tokens):
    """
    Imprime tokens de forma legível
    """
    token_list = [token.text for token in tokens]
    print("Tokens:", token_list)

def normalize_whitespace(text):
    """
    Substitui múltiplos espaços em branco por um único espaço e remove espaços extras no início/fim
    """
    return re.sub(r'\s+', ' ', text).strip()

def remove_html(text):
    """
    Remove tags HTML do texto
    """
    return BeautifulSoup(text, "html.parser").get_text()

def remove_special_characters(text):
    """
    Remove caracteres especiais, mantendo letras, números e espaços
    """
    return re.sub(r'[^A-Za-z0-9\s]', '', text)

def lemmatize_text(text, nlp):    
    """
    Reduz palavras à sua forma base
    """
    doc = nlp(text)
    lemmatized = ' '.join([token.lemma_ for token in doc])
    return lemmatized

def remove_stopwords(tokenized_text):
    """
    Mantém apenas palavras importantes
    """
    meaningful_words = [token.text for token in tokenized_text if not token.is_stop]
    return " ".join(meaningful_words)

def build_symspell():
    sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)

    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
    ok1 = sym_spell.load_dictionary(dictionary_path, 0, 1)
    ok2 = sym_spell.load_bigram_dictionary(bigram_path, 0, 2)

    print("Dictionary loaded:", ok1)
    print("Bigram loaded:", ok2)

    return sym_spell

def handle_typos(text, sym_spell):
    """
    Corrige erros de digitação (não deve ser usado no modelo de transformers)
    """
    corrected_words = []

    for word in text.split():

        clean = word.strip(string.punctuation)

        suggestions = sym_spell.lookup(clean.lower(),
                                       Verbosity.CLOSEST,
                                       max_edit_distance=2)

        corrected = suggestions[0].term if suggestions else clean

        corrected_words.append(corrected)

    return " ".join(corrected_words)

if __name__ == "__main__":
    text =     """
            This text has      extra spaces, <br> HTML </br>, 
            CAPTITALIZATION.   and special characters!!!
            Testing lemmatization: running, ran, easily farther and further.    
            Tseting typos: teh, recieve, adress, charcter.
            """
    nlp = en_core_web_sm.load()

    sym_spell = build_symspell()

    tokenized = tokenize(text, nlp)
    print("versão tokenizada:")
    print_tokenized(tokenized)
    print()

    normalized = normalize_whitespace(text)
    print(f"versão normalizada: {normalized}\n")

    no_html = remove_html(text)
    print(f"versão sem HTML: {no_html}\n")

    no_specials = remove_special_characters(text)
    print(f"versão sem caracteres especiais: {no_specials}\n")

    lemmatized = lemmatize_text(text, nlp)
    print(f"versão lematizada: {lemmatized}\n")
    
    no_stopwords = remove_stopwords(tokenized)
    print(f"versão sem stopwords: {no_stopwords}\n")


    no_typos = handle_typos(text, sym_spell=sym_spell)
    print(f"versão sem erros de digitação: {no_typos}\n")