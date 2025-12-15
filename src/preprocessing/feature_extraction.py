import numpy as np

def add_features(df, input_column='text'):
    """
    Adiciona features ao df
    """
    dataframe = df.copy()
    
    dataframe['length'] = dataframe[input_column].astype(str).apply(len)
    dataframe['word_count'] = dataframe[input_column].astype(str).apply(lambda x: len(x.split()))
    dataframe['mean_word_length'] = df[input_column].map(lambda rev: np.mean([len(word) for word in rev.split()]))
    
    return dataframe


