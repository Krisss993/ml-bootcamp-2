
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
