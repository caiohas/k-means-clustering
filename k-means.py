# %% [markdown]
# ### Instalando os pacotes necessários

# %%
!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install seaborn
!pip install plotly
!pip install scipy
!pip install scikit-learn
!pip install pingouin

# %% [markdown]
# ### Importação das bibliotecas necessárias

# %%
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
import numpy as np
import pyodbc

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import scipy.stats as stats
from scipy.stats import zscore
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pingouin as pg
import plotly.express as px 
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'

# %% [markdown]
# ### Leitura dos dados

# %%
# Objetivo: agrupar os clientes que receberam e-mail marketing
# Analisar os grupos de clientes por meio das suas interações com os envios

dados = pd.read_excel(r'C:\Users\caio.hsouza\OneDrive - MRV (1)\Área de Trabalho\knn linkedin\dados_emails.xlsx')
dados.head()

# %%
dados.info()

# %%
# Primeiramente, vamos selecionar apenas os campos das variáveis que serão utilizadas

cluster = dados[['qtd_emails_recebidos', 'qtd_abertura_email', 'qtd_click_email']]
cluster = cluster.fillna(0)
cluster = cluster.apply(pd.to_numeric, errors='coerce')
cluster

# %%
# Obtendo as estatísticas descritivas das variáveis
# Caso as variáveis sejam muito "diferentes" entre si, é válido padronizar o dataset para análise

tab_descritivas = cluster.describe().T
tab_descritivas

# %%
# Gráfico 3D das observações

fig = px.scatter_3d(cluster, 
                    x='qtd_emails_recebidos', 
                    y='qtd_abertura_email', 
                    z='qtd_click_email')
fig.show()

# %%
# Identificação da quantidade de clusters (Método Elbow)

elbow = []
K = range(1,11) # ponto de parada pode ser parametrizado manualmente
for k in K:
    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=100).fit(cluster)
    elbow.append(kmeanElbow.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, elbow, marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.xticks(range(1,11)) # ajustar range
plt.ylabel('WCSS', fontsize=16)
plt.title('Método de Elbow', fontsize=16)
plt.show()

# %%
# Identificação da quantidade de clusters (Método da Silhueta)

silhueta = []
I = range(2,11) # ponto de parada pode ser parametrizado manualmente
for i in I: 
    kmeansSil = KMeans(n_clusters=i, init='random', random_state=100).fit(cluster)
    silhueta.append(silhouette_score(cluster, kmeansSil.labels_))

plt.figure(figsize=(16,8))
plt.plot(range(2, 11), silhueta, color = 'purple', marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.ylabel('Silhueta Média', fontsize=16)
plt.title('Método da Silhueta', fontsize=16)
plt.axvline(x = silhueta.index(max(silhueta))+2, linestyle = 'dotted', color = 'red') 
plt.show()


# %%
# Utilizando o método K-Means
# Vamos considerar 3 clusters, considerando as evidências anteriores!

kmeans_final = KMeans(n_clusters = 3, init = 'random', random_state=100).fit(cluster)

# Gerando a variável para identificarmos os clusters gerados

kmeans_clusters = kmeans_final.labels_
cluster['cluster_kmeans'] = kmeans_clusters
cluster['cluster_kmeans'] = cluster['cluster_kmeans'].astype('category')


# %%
cluster

# %%
# Interpretação do output:

## cluster_kmeans MS: indica a variabilidade entre grupos
## Within MS: indica a variabilidade dentro dos grupos
## F: estatística de teste (cluster_kmeans MS / Within MS)
## p-unc: p-valor da estatística F
## se p-valor < 0.05: pelo menos um cluster apresenta média estatisticamente diferente dos demais

# qtd_emails_recebidos
pg.anova(dv='qtd_emails_recebidos', 
         between='cluster_kmeans', 
         data=cluster,
         detailed=True).T

# qtd_abertura_email
pg.anova(dv='qtd_abertura_email', 
         between='cluster_kmeans', 
         data=cluster,
         detailed=True).T

 # qtd_click_email
pg.anova(dv='qtd_click_email', 
         between='cluster_kmeans', 
         data=cluster,
         detailed=True).T

# %%
# Gráfico 3D dos clusters

fig = px.scatter_3d(cluster, 
                    x='qtd_emails_recebidos', 
                    y='qtd_abertura_email', 
                    z='qtd_click_email',
                    color='cluster_kmeans')
fig.show()

# %%
# Estatísticas descritivas por grupo

cluster_grupo = cluster.groupby(by=['cluster_kmeans'])

tab_desc_grupo = cluster_grupo.describe().T
tab_desc_grupo

# %%
cluster

# %%
cluster_mapping = {
    0: 'hotlead',
    1: 'saturado',
    2: 'inicio_jornada'
}


cluster['status'] = cluster['cluster_kmeans'].map(cluster_mapping).fillna('')
cluster

# %%
fig = px.scatter_3d(cluster, 
                    x='qtd_emails_recebidos', 
                    y='qtd_abertura_email', 
                    z='qtd_click_email',
                    color='status')  # Usando a coluna 'status' para colorir
fig.show()


