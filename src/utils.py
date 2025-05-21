import pandas as pd
import numpy as np

from scipy.stats import chi2_contingency


def gerar_metadados(df, ids, targets, orderby = 'PC_NULOS'):
    """
    Esta função retorna uma tabela com informações descritivas sobre um DataFrame.

    Parâmetros:
    - df: DataFrame que você quer descrever.
    - ids: Lista de colunas que são identificadores.
    - targets: Lista de colunas que são variáveis alvo.

    Retorna:
    Um DataFrame com informações sobre o df original.
    """
   
    summary = pd.DataFrame({
        'USO_FEATURE': ['ID' if col in ids else 'Target' if col in targets else 'Explicativa' for col in df.columns],
        'QT_NULOS': df.isnull().sum(),
        'PC_NULOS': round((df.isnull().sum() / len(df))* 100,2),
        'CARDINALIDADE': df.nunique(),
        'TIPO_FEATURE': df.dtypes
    })

    summary_sorted = summary.sort_values(by=orderby, ascending=False)
    summary_sorted = summary_sorted.reset_index()
    # Renomeando a coluna 'index' para 'FEATURES'
    summary_sorted = summary_sorted.rename(columns={'index': 'FEATURE'})
    return summary_sorted


def tratar_multicolinearidade(df, limiar=0.9):
    """
    Identifica pares de variáveis com alta correlação e sugere remoção
    da coluna com mais nulos. Em caso de empate, remove a de menor cardinalidade.

    Parâmetros:
    - df: DataFrame pandas
    - limiar: float, valor mínimo de correlação para considerar alta correlação

    Retorna:
    - DataFrame com os pares correlacionados
    - Lista de colunas sugeridas para remoção
    """

    # Calcular matriz de correlação
    corr_matrix = df.corr().abs()

    # Pegar apenas o triângulo superior
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Encontrar pares altamente correlacionados
    pares = (
        upper.stack()
        .reset_index()
        .rename(columns={'level_0': 'Variável 1', 'level_1': 'Variável 2', 0: 'Correlação'})
    )

    pares_correlacionados = pares[pares['Correlação'] >= limiar]

    # Contar nulos e cardinalidade
    nulos = df.isnull().sum()
    cardinalidade = df.nunique()

    colunas_para_remover = []

    for _, row in pares_correlacionados.iterrows():
        var1, var2 = row['Variável 1'], row['Variável 2']

        nulos1 = nulos[var1]
        nulos2 = nulos[var2]

        if nulos1 > nulos2:
            colunas_para_remover.append(var1)
        elif nulos2 > nulos1:
            colunas_para_remover.append(var2)
        else:
            # Se a quantidade de nulos for igual, usar cardinalidade como critério
            if cardinalidade[var1] <= cardinalidade[var2]:
                colunas_para_remover.append(var1)
            else:
                colunas_para_remover.append(var2)

    colunas_para_remover = list(set(colunas_para_remover))

    return pares_correlacionados.sort_values(by="Correlação", ascending=False), colunas_para_remover


def encontrar_colunas_duplicadas(df):
    """
    Compara colunas par à par identificando se as colunas são iguais.

    Parâmetros:
    - df: DataFrame pandas

    Retorna:
    - Tupla com os pares de colunas identicas
    """
    colunas_duplicadas = set()
    colunas = df.columns

    for i in range(len(colunas)):
        for j in range(i + 1, len(colunas)):
            col1 = colunas[i]
            col2 = colunas[j]

            # Verifica se os valores são iguais, considerando que NaN == NaN
            iguais = df[[col1, col2]].isna().all(axis=1) | (df[col1] == df[col2])

            if iguais.all():
                colunas_duplicadas.add((col1, col2))

    return colunas_duplicadas


def cramers_v(x, y):
    """
    Esta função calcula a correlação entre duas variáveis categóricas

    Parâmetros: 
    - x: Série categórica
    - y: Série categórica

    Retorna:
    - Valor da correlação entre as variáveis comparadas
    """

    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def matriz_cramers_v(df, colunas_categoricas):
    """
    Calcula a matriz de Cramér's V para um conjunto de colunas categóricas.

    Parâmetros:
    - df: DataFrame pandas
    - colunas_categoricas: Lista com as colunas categóricas

    Retorna:
    - Matriz de Correlação das variáveis categóricas
    """
    n = len(colunas_categoricas)
    matriz = pd.DataFrame(np.zeros((n, n)), 
                          index=colunas_categoricas, 
                          columns=colunas_categoricas)

    for i in range(n):
        for j in range(i, n):
            col1 = colunas_categoricas[i]
            col2 = colunas_categoricas[j]
            valor = cramers_v(df[col1], df[col2])
            matriz.loc[col1, col2] = valor
            matriz.loc[col2, col1] = valor  # simetria

    return matriz

def tratar_multicolinearidade_cat(df, colunas_categoricas, limiar=0.9):
    """
    Adaptação para variáveis categóricas.

    Identifica pares de variáveis com alta correlação e sugere remoção
    da coluna com mais nulos. Em caso de empate, remove a de menor cardinalidade.

    Parâmetros:
    - df: DataFrame pandas
    - limiar: float, valor mínimo de correlação para considerar alta correlação

    Retorna:
    - DataFrame com os pares correlacionados
    - Lista de colunas sugeridas para remoção
    """

    # Calcular matriz de correlação
    corr_matrix = matriz_cramers_v(df,colunas_categoricas)

    # Pegar apenas o triângulo superior
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Encontrar pares altamente correlacionados
    pares = (
        upper.stack()
        .reset_index()
        .rename(columns={'level_0': 'Variável 1', 'level_1': 'Variável 2', 0: 'Correlação'})
    )

    pares_correlacionados = pares[pares['Correlação'] >= limiar]

    # Contar nulos e cardinalidade
    nulos = df.isnull().sum()
    cardinalidade = df.nunique()

    colunas_para_remover = []

    for _, row in pares_correlacionados.iterrows():
        var1, var2 = row['Variável 1'], row['Variável 2']

        nulos1 = nulos[var1]
        nulos2 = nulos[var2]

        if nulos1 > nulos2:
            colunas_para_remover.append(var1)
        elif nulos2 > nulos1:
            colunas_para_remover.append(var2)
        else:
            # Se a quantidade de nulos for igual, usar cardinalidade como critério
            if cardinalidade[var1] <= cardinalidade[var2]:
                colunas_para_remover.append(var1)
            else:
                colunas_para_remover.append(var2)

    colunas_para_remover = list(set(colunas_para_remover))

    return pares_correlacionados.sort_values(by="Correlação", ascending=False), colunas_para_remover