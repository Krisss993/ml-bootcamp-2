
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
import plotly.express as px
import plotly.offline as pyo
import plotly.figure_factory as ff
import plotly.graph_objects as go


from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.plotting import plot_decision_regions

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn import datasets

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets.fashion_mnist import load_data
from sklearn.cluster import KMeans

#############

from numpy.linalg import norm
from sklearn.datasets import make_blobs
import random


from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
#########################
np.random.seed(41)


data = make_blobs(n_samples=10, centers=2, cluster_std=1.0, center_box=(-8.0, 8.0), random_state=42)[0]
df = pd.DataFrame(data, columns=['x1', 'x2'])
df.head()


plt.figure(figsize=(14, 7))
plt.scatter(data[:,0], data[:,1])

for label, x, y in zip(range(1, 11), data[:, 0], data[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(-3, 3), textcoords='offset points', ha='right', va='bottom')
plt.title('Grupowanie hierarchiczne')
plt.show()






# przeprowadzenie grupowania hierarchicznego
linked = linkage(data)

# wizualizacja grupowania hierarchicznego jako dendrogram
plt.figure(figsize=(14, 7))
dendrogram(linked, orientation='top', labels=range(1, 11), distance_sort='descending', show_leaf_counts=True)
plt.title('Grupowanie hierarchiczne - dendrogram')
plt.show()


plt.figure(figsize=(14, 7))
dendrogram(linked, orientation='right', labels=range(1, 11), distance_sort='descending', show_leaf_counts=True)
plt.title('Grupowanie hierarchiczne - dendrogram')
plt.show()



# bottom-up approach


cluster = AgglomerativeClustering(n_clusters=2)
cluster.fit_predict(data)


df = pd.DataFrame(data, columns=['x1', 'x2'])
df['cluster'] = cluster.labels_

fig = px.scatter(df, 'x1', 'x2', 'cluster', width=950, height=500, template='plotly_dark',
                 title='Grupowanie hierarchiczne', color_continuous_midpoint=0.6)
fig.update_traces(marker_size=12)
pyo.plot(fig)










# Porównanie metryk (euklidesowa, Manhattan, kosinusowa)



data = make_blobs(n_samples=1000, centers=4, cluster_std=1.5, center_box=(-8.0, 8.0), random_state=42)[0]
df = pd.DataFrame(data, columns=['x1', 'x2'])

px.scatter(df, 'x1', 'x2', width=950, height=500, title='Grupowanie hierarchiczne', template='plotly_dark')








cluster_euclidean = AgglomerativeClustering(n_clusters=4)
cluster_euclidean.fit_predict(data)

df_euclidean = pd.DataFrame(data, columns=['x1', 'x2'])
df_euclidean['cluster'] = cluster_euclidean.labels_
fig = px.scatter(df_euclidean, 'x1', 'x2', 'cluster', width=950, height=500, template='plotly_dark',
                 title='Grupowanie hierarchiczne - metryka euklidesowa', color_continuous_midpoint=0.6)
pyo.plot(fig)










cluster_manhattan = AgglomerativeClustering(n_clusters=4, linkage='complete',metric='manhattan')
cluster_manhattan.fit_predict(data)

df_manhattan = pd.DataFrame(data, columns=['x1', 'x2'])
df_manhattan['cluster'] = cluster_manhattan.labels_
fig = px.scatter(df_manhattan, 'x1', 'x2', 'cluster', width=950, height=500, template='plotly_dark',
                 title='Grupowanie hierarchiczne - metryka Manhattan', color_continuous_midpoint=0.6)
pyo.plot(fig)











cluster_cosine = AgglomerativeClustering(n_clusters=4, metric='cosine', linkage='complete')
cluster_cosine.fit_predict(data)

df_cosine = pd.DataFrame(data, columns=['x1', 'x2'])
df_cosine['cluster'] = cluster_cosine.labels_
fig = px.scatter(df_cosine, 'x1', 'x2', 'cluster', width=950, height=500, template='plotly_dark',
                 title='Grupowanie hierarchiczne - metryka kosinusowa', color_continuous_midpoint=0.6)
pyo.plot(fig)







from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=3, column_titles=['euclidean', 'Manhattan', 'cosine'])

fig1 = px.scatter(df_euclidean, 'x1', 'x2', 'cluster')['data'][0]
fig2 = px.scatter(df_manhattan, 'x1', 'x2', 'cluster')['data'][0]   
fig3 = px.scatter(df_cosine, 'x1', 'x2', 'cluster')['data'][0]                         

fig.add_trace(fig1, row=1, col=1)
fig.add_trace(fig2, row=1, col=2)
fig.add_trace(fig3, row=1, col=3)
fig.update_layout(template='plotly_dark')
pyo.plot(fig)
