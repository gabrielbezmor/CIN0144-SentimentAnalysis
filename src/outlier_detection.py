import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

"""
Detecção de Outliers - Múltiplos Métodos
Comparar resultados de diferentes técnicas para escolher a mais apropriada.
"""

def zscore_outliers(series, threshold=3):
    """
    Z-score method: identifica outliers com |z-score| > threshold.
    - threshold=3: ~99.7% dos dados em distribuição normal (extremos)
    - threshold=2: ~95% (mais sensível)
    - threshold=2.5: balanço entre sensibilidade e especificidade
    
    Retorna: (Series de outliers, máscara booleana, z-scores)
    """
    z_scores = np.abs(stats.zscore(series.dropna()))
    mask = z_scores > threshold
    outliers = series.dropna().iloc[mask]
    return outliers.sort_values(), mask, z_scores

def iqr_outliers(series, multiplier=1.5):
    """
    Interquartile Range (IQR) method: padrão em boxplots.
    - multiplier=1.5: outliers padrão (default Tukey)
    - multiplier=1.0: mais sensível
    - multiplier=3.0: apenas extremos
    
    Retorna: (Series de outliers, máscara booleana, (lower, upper))
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    mask = (series < lower) | (series > upper)
    outliers = series[mask]
    return outliers.sort_values(), mask, (lower, upper)

def isolation_forest_outliers(series, contamination=0.05):
    """
    Isolation Forest: modelo de ML não-paramétrico.
    Bom para dados multi-dimensionais e distribuições não-normais.
    - contamination: proporção esperada de outliers (0.05 = 5%)
    
    Retorna: (Series de outliers, máscara booleana, scores de anomalia)
    """
    if len(series) < 2:
        return series.iloc[:0], np.zeros(len(series), dtype=bool), np.array([])
    
    X = series.values.reshape(-1, 1)
    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    predictions = model.fit_predict(X)
    anomaly_scores = model.score_samples(X)
    
    mask = predictions == -1  # -1 = outlier, 1 = inlier
    outliers = series[mask]
    return outliers.sort_values(), mask, anomaly_scores

def local_outlier_factor_outliers(series, n_neighbors=100, contamination=0.05):
    """
    Local Outlier Factor (LOF): baseado em densidade local.
    Detecta pontos em regiões menos densas que seus vizinhos.
    - n_neighbors: número de vizinhos para comparação
    - contamination: proporção esperada de outliers
    
    Retorna: (Series de outliers, máscara booleana, LOF scores)
    """
    if len(series) < n_neighbors + 1:
        return series.iloc[:0], np.zeros(len(series), dtype=bool), np.array([])
    
    X = series.values.reshape(-1, 1)
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    predictions = model.fit_predict(X)
    lof_scores = model.negative_outlier_factor_
    
    mask = predictions == -1
    outliers = series[mask]
    return outliers.sort_values(), mask, lof_scores

def compare_outlier_methods(series, method_names=None):
    """
    Compara todos os métodos e retorna um DataFrame resumido.
    Útil para visualizar qual método detecta mais/menos outliers.
    
    Retorna: DataFrame com colunas = métodos e uma coluna 'consensus'
             que marca pontos detectados por 2+ métodos.
    """
    methods = {
        'zscore': lambda s: zscore_outliers(s, threshold=2.5)[1],
        'iqr': lambda s: iqr_outliers(s, multiplier=1.5)[1],
        'isolation_forest': lambda s: isolation_forest_outliers(s, contamination=0.05)[1],
        'lof': lambda s: local_outlier_factor_outliers(s, contamination=0.05)[1],
    }
    
    results = {}
    for name, method in methods.items():
        results[name] = method(series)
    
    results_df = pd.DataFrame(results, index=series.index)
    results_df['consensus'] = results_df.sum(axis=1) >= 2  # outlier em 2+ métodos
    
    return results_df


if __name__ == "__main__":
    # Teste rápido com dados sintéticos
    np.random.seed(42)
    data = pd.Series(np.concatenate([
        np.random.normal(100, 15, 200),  # distribuição normal
        np.array([500, 510, 520])  # 3 outliers extremos
    ]))
    
    print("=== Z-Score (threshold=3) ===")
    z_out, z_mask, z_scores = zscore_outliers(data, threshold=3)
    print(f"Outliers: {len(z_out)}\n{z_out.values}\n")
    
    print("=== IQR (multiplier=1.5) ===")
    iqr_out, iqr_mask, (lower, upper) = iqr_outliers(data)
    print(f"Outliers: {len(iqr_out)}, lower={lower:.1f}, upper={upper:.1f}\n{iqr_out.values}\n")
    
    print("=== Isolation Forest ===")
    if_out, if_mask, if_scores = isolation_forest_outliers(data, contamination=0.05)
    print(f"Outliers: {len(if_out)}\n{if_out.values}\n")
    
    print("=== Local Outlier Factor ===")
    lof_out, lof_mask, lof_scores = local_outlier_factor_outliers(data, contamination=0.05)
    print(f"Outliers: {len(lof_out)}\n{lof_out.values}\n")
    
    print("=== Consensus (2+ métodos) ===")
    consensus = compare_outlier_methods(data)
    consensus_outliers = data[consensus['consensus']]
    print(f"Consenso outliers: {len(consensus_outliers)}\n{consensus_outliers.values}")