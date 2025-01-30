import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic_settings import BaseSettings
import plotly.express as px
import plotly.figure_factory as ff
import scipy.stats as stats
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor  # Importando o regressor de Random Forest

# Título da aplicação
st.title('Análise de Dados - Boston Housing')

# Descrição sobre o dataset
st.write("""
Este aplicativo carrega o conjunto de dados de Boston Housing.
O objetivo é explorar e visualizar informações sobre o conjunto de dados.
""")

# Carregando a base de dados
link = 'https://ocw.mit.edu/courses/15-071-the-analytics-edge-spring-2017/d4332a3056f44e1a1dec9600a31f21c8_boston.csv'
boston = pd.read_csv(link)

# Exibindo o DataFrame completo
st.write('**Dataset Completo**:')
st.write(boston)

# Criar DataFrame Pandas
data = pd.DataFrame(data=boston)

# Exibir as primeiras linhas do DataFrame
st.write('**Primeiras Linhas do Dataset**:')
st.write(data.head())

# Descrever o DataFrame
st.write('**Descrição do Dataset**:')
st.write(data.describe())

# Exibir um gráfico de correlação entre variáveis
st.write('**Correlação entre Variáveis**:')

# Selecionar apenas as colunas numéricas
data_numeric = data.select_dtypes(include=[np.number])

# Calcular a correlação
corr = data_numeric.corr()

# Plotando a matriz de correlação com Seaborn
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
st.pyplot(plt)

# Exibir valores nulos em cada coluna
st.write('**Valores Nulos em Cada Coluna:**')
null_values = data.isnull().sum()
st.write(null_values)

# Remover a coluna TOWN
data = data.drop(['TOWN'], axis=1)

# Exibir o DataFrame atualizado
st.write("**DataFrame Após Remover a Coluna 'TOWN'**:")
st.write(data)

# Exibir a matriz de correlação
correlacoes = data.corr()
st.write("**Matriz de Correlação:**")
st.write(correlacoes)

# Exibir o gráfico de calor da matriz de correlação
st.write("**Gráfico de Calor da Matriz de Correlação:**")
plt.figure(figsize=(10, 8))
sns.heatmap(correlacoes, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
st.pyplot(plt)

# Gráfico de dispersão entre RM e MEDV
st.write("**Gráfico de Dispersão entre RM e MEDV:**")
fig = px.scatter(data, x="RM", y="MEDV", title="Relação entre RM e MEDV")
st.plotly_chart(fig)

# Estatísticas descritivas da coluna RM
st.write("**Estatísticas Descritivas da Coluna RM:**")
rm_stats = data.RM.describe()
st.write(rm_stats)

# Distribuição da variável RM
st.write("**Distribuição da Variável RM (Número de Quartos por Residência):**")
labels = ['Distribuição da variável RM (número de quartos)']
fig = ff.create_distplot([data.RM], labels, bin_size=0.2)
st.plotly_chart(fig)

# Calcular a assimetria da variável MEDV
skewness = stats.skew(data.MEDV)
st.write(f"A assimetria (skewness) da variável MEDV é: {skewness:.2f}")

# Criar o histograma de MEDV
fig = px.histogram(data, x="MEDV", nbins=50, opacity=0.50)
st.plotly_chart(fig)

# Converter RM para inteiro
data.RM = data.RM.astype(int)
st.write("Após a conversão:")
st.write(data[['RM']])

# Categorizar valores de RM
categorias = []
for valor in data.RM:
    if valor <= 4:
        categorias.append('Pequeno')
    elif valor < 7:
        categorias.append('Media')
    else:
        categorias.append("Grande")

# Adicionar a coluna 'Categoria_RM' ao DataFrame
data['Categoria_RM'] = categorias
st.write("**DataFrame com a Nova Coluna 'Categoria_RM':**")
st.write(data[['RM', 'Categoria_RM']])


# Divisão de dados em treinamento e teste
y = data['MEDV']
x = data.drop(['TOWN', 'TRACT', 'LAT', 'LON', 'RAD', 'TAX', 'MEDV', 'DIS', 'AGE', 'ZN', 'Categoria_RM'], axis=1, errors='ignore')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

# Modelo Baseline (valores médios por categoria)
predicoes = []
for i in range(len(x_test)):
    valor_rm = x_test.iloc[i]['RM']  # Acessando o valor de RM na linha i
    categoria = categorias[int(valor_rm)]  # Converte o valor de RM para um índice inteiro
    predicoes.append(categoria)

# Mapeando as categorias de 'RM' para valores numéricos
categoria_map = {'Pequeno': 0, 'Media': 1, 'Grande': 2}
predicoes_numericas = [categoria_map[categoria] for categoria in predicoes]

# Exibir resultados
df_results = pd.DataFrame()
df_results['valor_real'] = y_test.values
df_results['valor_predito_baseline'] = predicoes
st.write(df_results.head(10))

# Calcular RMSE para o modelo baseline com valores numéricos
rmse_baseline = np.sqrt(mean_squared_error(y_test, predicoes_numericas))
st.write(f'RMSE do Modelo Baseline: {rmse_baseline:.2f}')

# Treinando o modelo de regressão linear
lin_model = LinearRegression()
lin_model.fit(x_train, y_train)
y_pred = lin_model.predict(x_test)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred))

st.write(f'RMSE do Modelo de Regressão Linear: {rmse_linear:.2f}')

# Resultados do modelo linear
df_results['valor_predito_reg_linear'] = lin_model.predict(x_test)
st.write(df_results.head(10))

# Modelo de Árvore de Decisão
tree_model = DecisionTreeRegressor()
tree_model.fit(x_train, y_train)
y_pred_tree = tree_model.predict(x_test)

# Resultados da Árvore de Decisão
df_results['valor_predito_arvore'] = y_pred_tree
st.write(df_results.head(10))

rmse_tree = np.sqrt(mean_squared_error(y_test, y_pred_tree))
st.write(f'RMSE do Modelo de Árvore de Decisão: {rmse_tree:.2f}')

# ... (seu código anterior)

# Modelo de Random Forest
rf_regressor = RandomForestRegressor()
rf_regressor.fit(x_train, y_train)
y_pred_rf = rf_regressor.predict(x_test)

# Resultados do Random Forest
df_results['valor_predito_rf'] = y_pred_rf
st.write(df_results.head(10))

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
st.write(f'RMSE do Modelo Random Forest: {rmse_rf:.2f}')

# Visualização dos resultados
fig = go.Figure()

fig.add_trace(go.Scatter(x=df_results.index, y=df_results.valor_real, name='Valor real'))
fig.add_trace(go.Scatter(x=df_results.index, y=df_results.valor_predito_baseline, name='Valor predito baseline'))
fig.add_trace(go.Scatter(x=df_results.index, y=df_results.valor_predito_reg_linear, name='Valor predito reg linear'))
fig.add_trace(go.Scatter(x=df_results.index, y=df_results.valor_predito_arvore, name='Valor predito arvore'))
fig.add_trace(go.Scatter(x=df_results.index, y=df_results.valor_predito_rf, name='Valor predito rf'))

# Ajustando o layout (opcional)
fig.update_layout(
    title="Comparação de Modelos",
    xaxis_title="Índice",
    yaxis_title="Valor",
    height=600  # Definindo uma altura específica para o gráfico
)

# Exibindo o gráfico no Streamlit com ajuste de largura automática
st.plotly_chart(fig, use_container_width=True)

