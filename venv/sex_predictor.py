import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input_file', action = 'store', dest = 'input_file', required = True, help = 'input file')
arguments = parser.parse_args()
input_file_name = arguments.input_file

#importação dos dados com pandas
dados = pd.read_csv('test_data_CANDIDATE.csv')

#Tratamento dos dados

dados['sex'] = dados['sex'].str.upper()


dados = dados.fillna(0)


#carregando dataset nas variaveis X e y para fazer o treinamento
X = dados.drop(columns = ['sex'])
y = dados['sex']

#treinando o modelo com os parametros encontrados com GridSearch
model_rf = RandomForestClassifier(random_state=0, max_features='auto', n_estimators= 100, max_depth=5)
model_rf.fit(X, y)

#carregando o novo arquivo de exemplo sem a coluna sexo
newsample = pd.read_csv(input_file_name)

#removendo os valores nulos
newsample = newsample.fillna(0)

#fazendo a predição do modelo no novo arquivo
out = model_rf.predict(newsample)

#exportando arquivo csv com a coluna sexo dos dados previstos no arquivo de exemplo
out = pd.DataFrame(out)
out.columns = ['sex']
pd.DataFrame(out).to_csv('newsample_PREDICTIONS_Miguel_Angelo_C._R._de_Lima.csv')
