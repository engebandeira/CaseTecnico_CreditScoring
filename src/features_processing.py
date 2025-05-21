import pandas as pd
import pickle


def tratamento_completo(df, aux_vars_path,target_var_path,num_vars_path,cat_vars_path,cat_imputer_path,num_imputer_path,cat_encoder_path,num_scaler_path):
    """
    Aplica um pipeline completo de tratamento de dados a um DataFrame, incluindo seleção de variáveis,
    imputação de nulos, codificação de variáveis categóricas e normalização de variáveis numéricas.

    Parâmetros
    ----------
    df : pandas.DataFrame
        DataFrame de entrada com os dados brutos.

    aux_vars_path : str
        Caminho para o arquivo `.pickle` contendo a lista de variáveis auxiliares.

    target_var_path : str
        Caminho para o arquivo `.pickle` contendo a lista de variáveis alvo.

    num_vars_path : str
        Caminho para o arquivo `.pickle` contendo a lista de variáveis numéricas.

    cat_vars_path : str
        Caminho para o arquivo `.pickle` contendo a lista de variáveis categóricas.

    cat_imputer_path : str
        Caminho para o arquivo `.pickle` contendo o dicionário de imputação de variáveis categóricas.

    num_imputer_path : str
        Caminho para o arquivo `.pickle` contendo o imputador de variáveis numéricas.

    cat_encoder_path : str
        Caminho para o arquivo `.pickle` contendo o codificador para variáveis categóricas.

    num_scaler_path : str
        Caminho para o arquivo `.pickle` contendo o normalizador para variáveis numéricas.

    Retorna
    -------
    pandas.DataFrame
        DataFrame tratado com variáveis imputadas, codificadas e normalizadas, pronto para modelagem.
    """

    with open(aux_vars_path,'rb') as f:
        aux_vars = pickle.load(f)

    with open(target_var_path,'rb') as f:
        target_vars = pickle.load(f)
    
    with open(num_vars_path,'rb') as f:
        num_vars = pickle.load(f)

    with open(cat_vars_path,'rb') as f:
        cat_vars = pickle.load(f)

    with open(cat_imputer_path,'rb') as f:
        cat_imputer = pickle.load(f)

    with open(num_imputer_path,'rb') as f:
        num_imputer = pickle.load(f)

    with open(cat_encoder_path,'rb') as f:
        cat_encoder = pickle.load(f)

    with open(num_scaler_path,'rb') as f:
        num_scaler = pickle.load(f)  


    # Selecionando Colunas 
    df = df.reindex(columns=[*aux_vars,
                             *target_vars,
                             *num_vars,
                             *cat_vars])
    
    
    # Imputando Nulos
    df[num_vars] = num_imputer.transform(df[num_vars])
    
    for col, valor in cat_imputer.items():
        df[col] = df[col].fillna(valor)
   
    # Dummy encoding
    dados_one_hot = cat_encoder.transform(df)
    dados_one_hot = pd.DataFrame(dados_one_hot,
                     columns=cat_encoder.get_feature_names_out(),
                     index=df.index)
    
    df = pd.concat([df,dados_one_hot],axis=1)
    df = df.drop(columns=cat_vars)

    # Normalização
    df[num_vars] = num_scaler.transform(df[num_vars])

    
    return df