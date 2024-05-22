
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
from sklearn.manifold import TSNE
from plotly.subplots import make_subplots
#########################

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')


plt.figure(figsize=(12,8))
for i in range(8):
    plt.subplot(240+i+1)
    plt.imshow(X_train[i], cmap='gray_r')
    plt.title(y_train[i], fontsize=17)
    plt.axis('off')
plt.show()

X_train = X_train[:10000]
y_train = y_train[:10000]
X_train.shape
X_train = X_train.reshape(-1,28*28)
X_test = X_test.reshape(-1,28*28)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)


# METODA PCA


pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_std)
# X_test_pca = pca.transform(X_test_std)
X_train_pca.shape



results = pd.DataFrame({'explained_variance_ratio':pca.explained_variance_ratio_})
results['cumulative'] = np.cumsum(results['explained_variance_ratio'])
results['component'] = results.index+1
results

fig = go.Figure(data=[go.Bar(x=results['component'], y=results['explained_variance_ratio'], name='explained_variance_ratio'),
                      go.Scatter(x=results['component'], y=results['cumulative'], name='cumulative')],
                layout=go.Layout(title='PCA - 3 components', width=950, template='plotly_dark'))
pyo.plot(fig)

X_train_pca_df = pd.DataFrame(data=np.c_[X_train_pca, y_train], columns=['pca1','pca2','pca3','class'])
X_train_pca_df['class'] = X_train_pca_df['class'].astype(str)
X_train_pca_df

fig = px.scatter_3d(data_frame=X_train_pca_df, x=X_train_pca_df.iloc[:,0], y=X_train_pca_df.iloc[:,1], z=X_train_pca_df.iloc[:,2], color='class', symbol='class')
pyo.plot(fig)


pca = PCA(n_components=2)
X_train_pca_2 = pca.fit_transform(X_train_std)


X_train_pca_df = pd.DataFrame(data=np.c_[X_train_pca_2, y_train], columns=['pca1','pca2','class'])
X_train_pca_df['class'] = X_train_pca_df['class'].astype(str)
X_train_pca_df


# METODA TSNE



tsne= TSNE(n_components=2, verbose=True)
X_train_tsne = tsne.fit_transform(X_train_std)


X_train_tsne_df = pd.DataFrame(data=np.c_[X_train_tsne, y_train], columns=['tsne_1', 'tsne_2', 'class'])
X_train_tsne_df['class'] = X_train_tsne_df['class'].astype(str)
X_train_tsne_df

fig = px.scatter(X_train_tsne_df, x='tsne_1', y='tsne_2', color='class', opacity=0.5, width=950, height=700,
           template='plotly_dark', title='TSNE - 2 components')
pyo.plot(fig)





fig = make_subplots(rows=1, cols=2, subplot_titles=['PCA', 't-SNE'], horizontal_spacing=0.03)

fig1 = px.scatter(X_train_pca_df, x='pca1', y='pca2', color='class', opacity=0.5)
fig2 = px.scatter(X_train_tsne_df, x='tsne_1', y='tsne_2', color='class', opacity=0.5)

for i in range(0, 10):
    fig.add_trace(fig1['data'][i], row=1, col=1)
    fig.add_trace(fig2['data'][i], row=1, col=2)
fig.update_layout(width=950, showlegend=False, template='plotly_dark')
pyo.plot(fig)















# METODA PCA, POZNIEJ TSNE


pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train_std)
X_train_pca.shape




tsne = TSNE(n_components=2, verbose=1)
X_train_tsne_50 = tsne.fit_transform(X_train_pca)

X_train_tsne_50_df = pd.DataFrame(data=np.c_[X_train_tsne_50, y_train], columns=['tsne1','tsne2','class'])
X_train_tsne_50_df['class'] = X_train_tsne_50_df['class'].astype('str')
X_train_tsne_50_df


fig = px.scatter(X_train_tsne_50_df, x='tsne1', y='tsne2', color='class', opacity=0.5, width=950, height=700,
           template='plotly_dark', title='t-SNE - 2 components after PCA')
pyo.plot(fig)



fig = make_subplots(rows=1, cols=3, subplot_titles=['PCA', 't-SNE', 't-SNE after PCA'], horizontal_spacing=0.03)
fig1 = px.scatter(X_train_pca_df, x='pca1', y='pca2', color='class', opacity=0.5)
fig2 = px.scatter(X_train_tsne_df, x='tsne_1', y='tsne_2', color='class', opacity=0.5)
fig3 = px.scatter(X_train_tsne_50_df, x='tsne1', y='tsne2', color='class', opacity=0.5)

for i in range(0, 10):
    fig.add_trace(fig1['data'][i], row=1, col=1)
    fig.add_trace(fig2['data'][i], row=1, col=2)
    fig.add_trace(fig3['data'][i], row=1, col=3)
fig.update_layout(width=950, height=450, showlegend=False, template='plotly_dark')
pyo.plot(fig)




tsne = TSNE(n_components=3, verbose=1)
X_train_tsne = tsne.fit_transform(X_train_pca)

X_train_tsne_df = pd.DataFrame(data=np.c_[X_train_tsne, y_train], columns=['tsne1', 'tsne2', 'tsne3', 'class'])
X_train_tsne_df['class'] = X_train_tsne_df['class'].astype('str')
X_train_tsne_df

fig = px.scatter_3d(X_train_tsne_df, x='tsne1', y='tsne2', z='tsne3', color='class',symbol='class', opacity=0.5, width=950, height=700,
           template='plotly_dark', title='TSNE - 3 components')
pyo.plot(fig)
