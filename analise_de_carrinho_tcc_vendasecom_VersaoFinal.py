# -*- coding: utf-8 -*-
"""Analise_de_Carrinho_TCC_VendasEcom.ipynb

Desenvolvido por: Rodrigo Garcia Zaroni - JAN/2024
"""

!pip install apyori

import numpy as np
import pandas as pd
import plotly.express as px
import apyori
from apyori import apriori
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

#Carregando os dados de vendas localizados na mesma pasta do script
df = pd.read_excel('Vendas_Ecom.xlsx')

count_row = df.shape[0]  # Número de Linhas
count_col = df.shape[1]  # Número de Colunas
print('Linhas:', count_row, ' com ', count_col, 'colunas')

# Exibir os nomes dos campos e seus tipos
print(df.dtypes)

# Anonimizando o campo e-mail , sem perder a referência para evitar a exposição dos dados do comprador
df['Email'] = df['Email'].apply(lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest())

# Selecionando as colunas desejadas para visualizar
selected_columns = df[['Email', 'order_id','item_sku','item_name','item_quantity','item_total','Subtotal','Total','order_status','Created at']].head()

# Salvando o conteúdo das colunas selecionadas e Email anonimizado para futuras consultas ou estudos posteriores
selected_columns.to_csv('selected_columns.csv', index=False)

# Convertendo o DataFrame em uma lista de dicionários
data_as_dicts = selected_columns.to_dict('records')

# Exibindo os dados em formato de tabela
print(tabulate(data_as_dicts, headers="keys", tablefmt="pretty"))

# Criando os gráficos

# Gráfico de barras Email x count de 'order_id' (TOP 10)
plt.figure(figsize=(12, 6))
top_emails = df['Email'].value_counts().head(10)
sns.barplot(x=top_emails.index, y=top_emails.values)
plt.xlabel('E-mail (Anonimizado)')
plt.ylabel('Contagem de order_id')
plt.title('TOP 10 de E-mails com Maior Contagem de order_id')
plt.xticks(rotation=90)
plt.show()

# Gráfico de barras item_sku x soma de item_quantity (TOP 10)
plt.figure(figsize=(12, 6))
top_skus = df.groupby('item_sku')['item_quantity'].sum().nlargest(10)
sns.barplot(x=top_skus.index, y=top_skus.values)
plt.xlabel('SKU do Item')
plt.ylabel('Soma de Quantidade de Item')
plt.title('TOP 10 de SKUs com Maior Soma de Quantidade de Item')
plt.xticks(rotation=75)
plt.show()

# Gráfico de barras para 'order_status' x contagem de 'order_id'
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x='order_status')
plt.xlabel('Status do Pedido')
plt.ylabel('Contagem de order_id')
plt.title('Contagem de order_id por Status do Pedido')
plt.xticks(rotation=45)

# Adicionando os números máximos acima das barras
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.show()

# Convertendo 'Created at' para o formato de data (considerando o fuso horário UTC)
df['Created at'] = pd.to_datetime(df['Created at'], errors='coerce', utc=True)

# Removendo linhas com valores de data nulos (caso haja)
df = df.dropna(subset=['Created at'])

# Criando uma série temporal de 'order_id' por ano/mês (YYYYMM)
df['YearMonth'] = df['Created at'].dt.strftime('%Y%m')

# Gráfico de linha para 'order_id' ao longo do tempo (ano/mês) - Contagem de order_id
plt.figure(figsize=(12, 6))
df.groupby('YearMonth')['order_id'].count().plot(kind='line', marker='o', label='Contagem de order_id')

# Gráfico de linha para a média da coluna 'Total' ao longo do tempo (ano/mês)
df.groupby('YearMonth')['Total'].mean().plot(kind='line', marker='o', label='Média de Total', linestyle='dashed')

plt.xlabel('Ano e Mês (YYYYMM)')
plt.ylabel('Valores')
plt.title('Série Temporal de Contagem de order_id e Média de Total por Ano e Mês')
plt.xticks(rotation=45)
plt.legend()
plt.show()

df['Total'].mean()

# Mantendo apenas as concluídas
df = df[df['order_status'] == 'completed']
df = df.dropna(subset=['item_name'])


#Ajustando SKU para char e evitar erros de compatibilidade de str e float
df['item_sku'] = df['item_sku'].astype(str)

print("Top 10 Vendas)")
x = df['item_name'].value_counts().sort_values(ascending=False)[:10]
fig = px.bar(x= x.index, y= x.values)
fig.update_layout(title_text= "Top 10 Produtos mais vendidos", xaxis_title= "Produtos", yaxis_title="Vendidos")
fig.show()

"""Preparando a lista para entrada no algorítimo  Apriori"""

# Agrupando os itens por transação
grouped = df.groupby('Email')['item_sku'].apply(list)

# Convertendo o DataFrame agrupado em uma lista de transações
transactions = list(grouped)

# Converter números inteiros para strings
transactions_as_strings = []

for transaction in transactions:
    transaction_as_string = [str(item) for item in transaction]
    transactions_as_strings.append(transaction_as_string)


# print(transactions)
print(transactions_as_strings)

"""Parâmetros utilizados na função apriori:
transactions: Este é o conjunto de dados de entrada para o algoritmo Apriori. Como mencionado anteriormente, deve ser uma lista de transações, onde cada transação é uma lista dos itens comprados.
min_support: O suporte mínimo é um dos principais parâmetros do algoritmo Apriori. Ele define o limite mínimo para considerar um itemset (conjunto de itens) como "frequente" no conjunto de dados. O suporte de um itemset é calculado como a proporção de transações no conjunto de dados que contêm esse itemset. Por exemplo, um min_support de 0.0005 significa que um itemset precisa aparecer em pelo menos 0.05% de todas as transações para ser considerado no processo de mineração de regras.
min_confidence: A confiança mínima é outro parâmetro crucial. É usado para medir a confiabilidade de uma regra de associação inferida. A confiança é a proporção de vezes que, se o lado esquerdo de uma regra (antecedente) aparece em uma transação, o lado direito da regra (consequente) também aparece. Um min_confidence de 0.05 significa que apenas as regras de associação com uma confiança de pelo menos 5% serão consideradas.
min_lift: O lift é uma métrica que compara a confiança de uma regra com a expectativa de que os itens apareçam juntos se fossem independentes. Um min_lift de 3 indica que você está interessado apenas em regras onde a probabilidade de ocorrência conjunta dos itens é pelo menos três vezes maior do que seria se fossem independentes.
max_length: Este parâmetro define o tamanho máximo dos itemsets a serem considerados. Por exemplo, um max_length de 2 significa que o algoritmo só considerará itemsets contendo no máximo dois itens. Isso é útil para limitar a complexidade das regras geradas e focar em relações mais simples.
target="rules": Este parâmetro indica o tipo de itemset que você está interessado em encontrar. Neste caso, "rules" significa que você está interessado em encontrar regras de associação, não apenas conjuntos frequentes de itens.
"""

# Aplicar o algoritmo Apriori para 2 produtos
# com suporte mínimo de 0.05% , um mínimo de confiança de 25% e um lift
rules = apriori(transactions_as_strings, min_support=(0.02/100), min_confidence=(10/100), min_lift=3, max_length=2, target="rules")
association_results = list(rules)
print(association_results[0])

# Lista para armazenar os resultados
results = []

for item in association_results:
    pair = item[0]
    items = [x for x in pair]

    # Adicionando os resultados à lista
    results.append({
        'Rule': f"{items[0]} -> {items[1]}",
        'Support': item[1],
        'Confidence': item[2][0][2],
        'Lift': item[2][0][3]
    })

    # Exibir os resultados
    print(f"Rule : {items[0]} -> {items[1]}")
    print(f"Support : {item[1]}")
    print(f"Confidence : {item[2][0][2]}")
    print(f"Lift : {item[2][0][3]}")
    print("=============================")

# Convertendo a lista em DataFrame
results_df = pd.DataFrame(results)

# Salvando o DataFrame em um arquivo CSV que será utilizado na API
results_df.to_csv('association_rulesEcom.csv', index=False)