import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


file_path = r"C:\Users\USUARIO\Downloads\Parte_pratica-Rafael_Resende-20241217T121431Z-001\segunda_questão\healthcare-dataset-stroke-data.csv"  # Substitua pelo caminho correto
data = pd.read_csv(file_path)

# trasformando as variaves em num.
data['smoking_score'] = data['smoking_status'].replace({'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 0})

data['age_gender_score'] = data['age']
data['age_gender_score'] += data.apply(
    lambda row: 1 if (row['gender'] == 'Female' and row['age'] < 55) or (row['gender'] == 'Male' and row['age'] < 60) else 0,
    axis=1
)

data['urban_score'] = data['Residence_type'].replace({'Urban': 1, 'Rural': 0})

# Selecionar variáveis de entrada para comparar
X = data[['smoking_score', 'age_gender_score', 'urban_score']]
y = data['stroke']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Treino
logistic_model = LogisticRegression(random_state=42, max_iter=1000)
logistic_model.fit(X_train, y_train)

# Buscar o id
def prever_avc_por_id(id_pessoa):
    pessoa = data[data['id'] == id_pessoa]
    if pessoa.empty:
        return "ID não encontrado."
    

    entrada = pessoa[['smoking_score', 'age_gender_score', 'urban_score']]
    
    #comparação das entradas
    probabilidade = logistic_model.predict_proba(entrada)[0][1]
    previsao = logistic_model.predict(entrada)[0]
    
    # Retornar o resultado
    if previsao == 1:
        return f"A pessoa com ID {id_pessoa} tem alta probabilidade de sofrer um AVC (Probabilidade: {probabilidade:.2f})."
    else:
        return f"A pessoa com ID {id_pessoa} tem baixa probabilidade de sofrer um AVC (Probabilidade: {probabilidade:.2f})."

id_pessoa = int(input("Digite o ID da pessoa: "))
print(prever_avc_por_id(id_pessoa))
