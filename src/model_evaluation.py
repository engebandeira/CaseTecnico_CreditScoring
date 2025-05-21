import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, roc_curve
from sklearn.metrics     import brier_score_loss
from sklearn.calibration import calibration_curve

def calculate_ks(y_true, y_score):
    """Calculate KS statistic."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return 100*max(tpr - fpr)

def calculate_gini(y_true, y_score):
    """Calculate Gini coefficient."""
    return (2 * roc_auc_score(y_true, y_score) - 1)*100


def calibration_metrics(y_true, y_prob, n_bins=10):

    fraction_of_positives, mean_predicted_value = calibration_curve(y_true,
                                                                    y_prob,
                                                                    n_bins=n_bins,
                                                                    strategy="uniform")
    # Determinar os tamanhos dos bins
    bin_indices = np.digitize(y_prob, bins=np.linspace(0, 1, n_bins + 1)) - 1
    bin_sizes = np.array([np.sum(bin_indices == i) for i in range(n_bins)])
    
    # Remover bins vazios para evitar divisão por zero
    valid_bins = bin_sizes > 0
    fraction_of_positives = fraction_of_positives[valid_bins]
    mean_predicted_value = mean_predicted_value[valid_bins]
    bin_sizes = bin_sizes[valid_bins]
    
    # Calcular o ECE
    total_samples = len(y_true)
    ece = np.sum((bin_sizes / total_samples) * np.abs(fraction_of_positives - mean_predicted_value))
    brier_score = brier_score_loss(y_true, y_prob)

    print(f"Brier Score: {brier_score:.4f}")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")

    # Plotar a reliability curve
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Modelo")
    plt.plot([0, 1], [0, 1], "k--", label="Perfeitamente Calibrado")  # Linha de referência
    plt.title("Reliability Curve (Curva de Calibração)")
    plt.xlabel("Probabilidade Predita")
    plt.ylabel("Fração de Positivos")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

def performance_metrics(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks   = 100*max(tpr - fpr)
    auc  = roc_auc_score(y_true, y_prob)
    gini = (2 * roc_auc_score(y_true, y_prob) - 1)*100

    return ks, auc, gini

def construct_metrics_table(df, group_column, score_column, target_column):
    table = (df.groupby(group_column)
                    .agg(min_score=(score_column, 'min'),
                        max_score=(score_column, 'max'),
                        event_rate=(target_column, 'mean'),
                        volume=(target_column, 'size'))
                    .reset_index())
    return table

def compare_performance(df_treino_true,df_treino_prob, df_val_true, df_val_prob):
    ks_treino, auc_treino, gini_treino = performance_metrics(df_treino_true,df_treino_prob)
    ks_val, auc_val, gini_val = performance_metrics(df_val_true,df_val_prob)

    metrics_df = pd.DataFrame({
        'Métrica': ['KS', 'AUC', 'Gini'],
        'Treino': [ks_treino, auc_treino, gini_treino],
        'Validação': [ks_val, auc_val, gini_val]})
    
    return metrics_df

def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    # Define os limites dos bins com base nos quantis da base expected
    breakpoints = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    
    # Calcula a proporção de elementos em cada bin para expected e actual
    exp_counts, _ = np.histogram(expected, bins=breakpoints)
    act_counts, _ = np.histogram(actual, bins=breakpoints)

    exp_pct = exp_counts / len(expected)
    act_pct = act_counts / len(actual)

    # Substitui proporções zero por valor pequeno para evitar log(0)
    exp_pct = np.where(exp_pct == 0, 1e-8, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-8, act_pct)

    # Calcula o PSI
    psi_values = (exp_pct - act_pct) * np.log(exp_pct / act_pct)
    return np.sum(psi_values)