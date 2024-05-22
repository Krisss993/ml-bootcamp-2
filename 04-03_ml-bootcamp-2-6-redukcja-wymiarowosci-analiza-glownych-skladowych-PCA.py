
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
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
#########################
np.random.seed(41)

raw_data = load_breast_cancer()
all_data = raw_data.copy()
data = all_data['data']
target = all_data['target']
data[:3]


scaler = StandardScaler()
data_std = scaler.fit_transform(data)

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_std)
data_pca[:5]


pca_2 = pd.DataFrame({'pca_1':data_pca[:,0], 'pca_2':data_pca[:,1],'class':target})
pca_2.rename(columns={'pca_1':'Benign','pca_2':'Malignant'}, inplace=True)
pca_2

results = pd.DataFrame(data={'explained_variance_ratio': pca.explained_variance_ratio_})
results['cumulative'] = results['explained_variance_ratio'].cumsum()
results['component'] = results.index + 1
results


fig = go.Figure(data=[go.Bar(x=results['component'], y=results['explained_variance_ratio'], name='explained_variance_ratio'),
                      go.Scatter(x=results['component'], y=results['cumulative'], name='cumulative')],
                layout=go.Layout(title='PCA - 2 components', width=950, template='plotly_dark'))
pyo.plot(fig)



fig = px.scatter(pca_2, 'Benign', 'Malignant', color=pca_2['class'], width=950, template='plotly_dark')
pyo.plot(fig)














pca = PCA(n_components=3)
data_pca = pca.fit_transform(data_std)

pca_3 = pd.DataFrame(data={'pca_1':data_pca[:,0],'pca_2':data_pca[:,1],'pca_3':data_pca[:,2]})
pca_3['class'] = target
pca_3
pca_3.replace(0, 'Benign', inplace=True)
pca_3.replace(1, 'Malignant', inplace=True)
pca_3.head()

results = pd.DataFrame(data={'explained_variance_ratio':pca.explained_variance_ratio_})
results['cumulative'] = np.cumsum(results['explained_variance_ratio'])
results['component'] = results.index+1
results

fig = go.Figure(data=[go.Bar(x=results['component'], y=results['explained_variance_ratio'], name='explained_variance_ratio'),
                      go.Scatter(x=results['component'], y=results['cumulative'], name='cumulative')],
                layout=go.Layout(title='PCA - 2 components', width=950, template='plotly_dark'))
pyo.plot(fig)



fig = px.scatter_3d(pca_3, x='pca_1', y='pca_2', z='pca_3', color='class', symbol='class', 
              opacity=0.7, size_max=10, width=950, template='plotly_dark')
pyo.plot(fig)







