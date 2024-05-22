
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
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons

import random

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
from plotly.subplots import make_subplots
#########################
np.random.seed(41)




blobs_data = make_blobs(n_samples=1000, cluster_std=0.7, random_state=24, center_box=(-4.0, 4.0))[0]
blobs = pd.DataFrame(blobs_data, columns=['x1', 'x2'])
fig = px.scatter(blobs, 'x1', 'x2', width=950, height=500, title='blobs data', template='plotly_dark')
pyo.plot(fig)







circle_data = make_circles(n_samples=1000, factor=0.5, noise=0.05)[0]
circle = pd.DataFrame(circle_data, columns=['x1', 'x2'])
fig = px.scatter(circle, 'x1', 'x2', width=950, height=500, title='circle data', template='plotly_dark')
pyo.plot(fig)





moons_data = make_moons(n_samples=1000, noise=0.05)[0]
moons = pd.DataFrame(moons_data, columns=['x1', 'x2'])
fig = px.scatter(moons, 'x1', 'x2', width=950, height=500, title='moons data', template='plotly_dark')
pyo.plot(fig)




random_data = np.random.rand(1500, 2)
random = pd.DataFrame(random_data, columns=['x1', 'x2'])
fig = px.scatter(random, 'x1', 'x2', width=950, height=500, title='random data', template='plotly_dark')
pyo.plot(fig)



plt.scatter(blobs.iloc[:,0], blobs['x2'])






fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.01)

kmeans = KMeans(n_clusters=3)
kmeans.fit(blobs_data)
clusters = kmeans.predict(blobs_data)
blobs['cluster'] = clusters
trace1 = px.scatter(blobs, 'x1', 'x2', 'cluster', width=800, height=500)['data'][0]
fig.add_trace(trace1, row=1, col=1)

agglo = AgglomerativeClustering(n_clusters=3,metric='euclidean')
clusters = agglo.fit_predict(blobs_data)
blobs['cluster'] = clusters
trace2 = px.scatter(blobs, 'x1', 'x2', 'cluster', width=800, height=500)['data'][0]
fig.add_trace(trace2, row=1, col=2)

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(blobs_data)
clusters = dbscan.labels_
blobs['cluster'] = clusters
trace3 = px.scatter(blobs, 'x1', 'x2', 'cluster', width=800, height=500)['data'][0]
fig.add_trace(trace3, row=1, col=3)

fig.update_layout(title='KMeans vs. Agglomerative Clustering vs. DBSCAN - blobs data', 
                  template='plotly_dark', coloraxis = {'colorscale':'viridis'})
pyo.plot(fig)




fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.01)

kmeans = KMeans(n_clusters=3)
#kmeans.fit_predict()
kmeans.fit(moons_data)
clusters = kmeans.predict(moons_data)
moons['cluster'] = clusters
trace1 = px.scatter(moons, 'x1', 'x2', 'cluster', width=800, height=500)['data'][0]
fig.add_trace(trace1, row=1, col=1)


agglo = AgglomerativeClustering(n_clusters=3, metric='euclidean')

clusters = agglo.fit_predict(moons_data)
moons['cluster'] = clusters
trace2 = px.scatter(moons, 'x1', 'x2', 'cluster', width=800, height=500)['data'][0]
fig.add_trace(trace2, row=1, col=2)



dbscan = DBSCAN(eps=0.1, min_samples=5)
clusters = dbscan.fit_predict(moons_data)
moons['cluster'] = clusters
trace3 = px.scatter(moons, 'x1', 'x2', 'cluster', width=800, height=500)['data'][0]
fig.add_trace(trace3, row=1, col=3)
fig.update_layout(title='KMeans vs. Agglomerative Clustering vs. DBSCAN - blobs data', 
                  template='plotly_dark', coloraxis = {'colorscale':'viridis'})
pyo.plot(fig)












fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.01)

kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(circle_data)
circle['cluster'] = clusters
trace1 = px.scatter(circle, 'x1', 'x2', 'cluster', width=800, height=500)['data'][0]
fig.add_trace(trace1, row=1, col=1)

agglo = AgglomerativeClustering(n_clusters=3,metric='euclidean')
clusters = agglo.fit_predict(circle_data)
circle['cluster'] = clusters
trace2 = px.scatter(circle, 'x1', 'x2', 'cluster', width=800, height=500)['data'][0]
fig.add_trace(trace2, row=1, col=2)

dbscan = DBSCAN(eps=0.2, min_samples=5)
clusters = dbscan.fit_predict(circle_data)
circle['cluster'] = clusters
trace3 = px.scatter(circle, 'x1', 'x2', 'cluster', width=800, height=500)['data'][0]
fig.add_trace(trace3, row=1, col=3)
fig.update_layout(title='KMeans vs. Agglomerative Clustering vs. DBSCAN - blobs data', 
                  template='plotly_dark', coloraxis = {'colorscale':'viridis'})
pyo.plot(fig)












fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.01)

kmeans = KMeans(n_clusters=4)
clusters = kmeans.fit_predict(random_data)
# transf = kmeans.fit_transform(random_data)
random['cluster'] = clusters
trace1 = px.scatter(random, 'x1', 'x2', 'cluster', width=800, height=500)['data'][0]
fig.add_trace(trace1, row=1, col=1)


agglo = AgglomerativeClustering(n_clusters=4, metric='euclidean')
clusters = agglo.fit_predict(random_data)
random['cluster'] = clusters
trace2 = px.scatter(random, 'x1', 'x2', 'cluster', width=800, height=500)['data'][0]
fig.add_trace(trace2, row=1, col=2)


dbscan = DBSCAN(eps=0.1, min_samples=5)
clusters = dbscan.fit_predict(random_data)
random['cluster'] = clusters
trace3 = px.scatter(random, 'x1', 'x2', 'cluster', width=800, height=500)['data'][0]
fig.add_trace(trace3, row=1, col=3)
fig.update_layout(title='KMeans vs. Agglomerative Clustering vs. DBSCAN - blobs data', 
                  template='plotly_dark', coloraxis = {'colorscale':'viridis'})
pyo.plot(fig)












































