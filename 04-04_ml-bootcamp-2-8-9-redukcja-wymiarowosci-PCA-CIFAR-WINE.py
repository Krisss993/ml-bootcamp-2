
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
from keras.datasets import mnist
from keras.datasets import cifar10
#########################


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

X_train = X_train[:5000]
y_train = y_train[:5000]

#scaler = StandardScaler()






targets = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

plt.imshow(X_train[1])
plt.title(targets[y_train[1][0]], fontsize=17)
plt.axis('off')
plt.show()

plt.figure(figsize=(12,8))
for i in range(8):
    plt.subplot(240+i+1)
    plt.imshow(X_train[i])
    plt.title(targets[y_train[i][0]])
plt.show()


X_train = X_train / 255.
X_test = X_test /255.

X_train.shape
X_train = X_train.reshape(-1,32*32*3)
X_train.shape

pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train)


results = pd.DataFrame(data = {'explained_variance_ratio':pca.explained_variance_ratio_})
results['cumulative'] = np.cumsum(results['explained_variance_ratio'])
results['component'] = results.index+1


fig = go.Figure(data=[go.Bar(x=results['component'], y=results['explained_variance_ratio'], name='explained_variance_ratio'),
                      go.Scatter(x=results['component'], y=results['cumulative'], name='cumulative')],
                layout=go.Layout(title='PCA - 3 components', width=950, template='plotly_dark'))
pyo.plot(fig)


X_train_pca_df = pd.DataFrame(data=np.c_[X_train_pca, y_train], columns=['pca_1', 'pca_2', 'pca_3', 'class'])
X_train_pca_df['class'] = X_train_pca_df['class'].map(targets)
X_train_pca_df.info()

fig = px.scatter_3d(X_train_pca_df, x='pca_1', y='pca_2', z='pca_3', color='class', 
              symbol='class', opacity=0.7, size_max=10, width=950, height=700,
              title='PCA - CIFAR dataset', template='plotly_dark')
pyo.plot(fig)



pca = PCA(n_components=0.90)
X_train_pca = pca.fit_transform(X_train)
pca.explained_variance_ratio_
pca.n_components_


results = pd.DataFrame(data={'explained_variance_ratio': pca.explained_variance_ratio_})
results['cumulative'] = results['explained_variance_ratio'].cumsum()
results['component'] = results.index + 1
results.head()


fig = go.Figure(data=[go.Bar(x=results['component'], y=results['explained_variance_ratio'], name='explained_variance_ratio'),
                      go.Scatter(x=results['component'], y=results['cumulative'], name='cumulative')],
                layout=go.Layout(title=f'PCA - {pca.n_components_} components', width=950, template='plotly_dark'))
pyo.plot(fig)






















######################################                                       WINE                                  ######################################   




df_raw = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df = df_raw.copy()
df.head()

data = df.iloc[:,1:]
target = df.iloc[:,0]


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
pca.explained_variance_ratio_

X_train_pca_df = pd.DataFrame(data={'pca_1':X_train_pca[:,0], 'pca_2':X_train_pca[:,1], 'pca_3':X_train_pca[:,2]})
X_train_pca_df['class'] = target
X_train_pca_df

fig = px.scatter_3d(data_frame=X_train_pca_df, x='pca_1', y='pca_2', z='pca_3', color='class', symbol='class')
pyo.plot(fig)

results = pd.DataFrame(data={'explained_variance_ratio':pca.explained_variance_ratio_})
results['cumulative'] = np.cumsum(results['explained_variance_ratio'])
results['component'] = results.index + 1
results

fig = go.Figure(data = [go.Bar(x=results['component'], y=results['explained_variance_ratio']), 
                        go.Scatter(x=results['component'], y=results['cumulative'])],
                layout=go.Layout(title='PCA - 3 components', width=950, template='plotly_dark'))
pyo.plot(fig)





































