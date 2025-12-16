"""
Funções para limpar o dataset como um todo, não o conteúdo 
específico das linhas.
"""

def remove_duplicates(dataframe):
    """
    Remove linhas duplicadas do dataframe.
    Retorna o dataframe sem duplicatas.
    """
    return dataframe.drop_duplicates(keep="first").reset_index(drop=True)