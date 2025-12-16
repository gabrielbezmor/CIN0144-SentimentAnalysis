import pandas as pd

"""
Funções para limpar o dataset como um todo, não o conteúdo 
específico das linhas.
"""

def remove_duplicates(dataframe):
    """
    Remove linhas duplicadas do dataframe.
    Retorna o dataframe sem duplicatas.
    """
    return dataframe.drop_duplicates().reset_index(drop=True)

def combine_data(train_path, test_path, val_path):
    print("Carregando arquivos")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_val = pd.read_csv(val_path)
    df_full = pd.concat([df_train,df_val,df_test], axis=0)
    return df_full