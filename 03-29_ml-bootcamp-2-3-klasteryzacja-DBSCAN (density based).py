
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

from sklearn.cluster import DBSCAN

#########################
np.random.seed(41)


data = make_blobs(n_samples=1000, centers=3, cluster_std=1.2, center_box=(-8.0, 8.0), random_state=42)[0]
df = pd.DataFrame(data, columns=['x1', 'x2'])
px.scatter(df, 'x1', 'x2', width=950, height=500, title='Klasteryzacja', template='plotly_dark')


cluster = DBSCAN(eps=0.5, min_samples=7)
cluster.fit(data)

cluster.labels_[:10]

df['cluster'] = cluster.labels_

fig = px.scatter(df, 'x1', 'x2', 'cluster', width=950, height=500, title='DBSCAN(eps=0.5, min_samples=5)', 
           template='plotly_dark', color_continuous_midpoint=0)
pyo.plot(fig)






data = make_blobs(n_samples=1000, centers=4, cluster_std=1.2, center_box=(-8.0, 8.0), random_state=43)[0]
df = pd.DataFrame(data, columns=['x1', 'x2'])
px.scatter(df, 'x1', 'x2', width=950, height=500, title='DBSCAN', template='plotly_dark')




cluster = DBSCAN(eps=0.8, min_samples=10)
cluster.fit(data)
df['cluster'] = cluster.labels_

fig = px.scatter(df, 'x1', 'x2', 'cluster', width=950, height=500, title='DBSCAN(eps=0.5, min_samples=5)', 
           template='plotly_dark', color_continuous_midpoint=0)
pyo.plot(fig)



df['cluster'].value_counts()

DBSCAN?
