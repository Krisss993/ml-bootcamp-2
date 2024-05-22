

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


##########
from sklearn.datasets import make_blobs

from sklearn.neighbors import LocalOutlierFactor 

from sklearn.ensemble import IsolationForest


##########

data = pd.read_csv('https://storage.googleapis.com/esmartdata-courses-files/ml-course/factory.csv')
data.head()


data.describe()

fig = px.scatter(data, x='item_length', y='item_width', width=950, template='plotly_dark', title='Isolation Forest')
pyo.plot(fig)

# contamination in [0, 0.05]
outlier = IsolationForest(n_estimators=100, contamination=0.05)
outlier.fit(data)



y_pred = outlier.predict(data)
y_pred[:30]


data['outlier_flag'] = y_pred
fig = px.scatter(data, x='item_length', y='item_width', color='outlier_flag', width=950, template='plotly_dark',
           color_continuous_midpoint=-1, title='Isolation Forest')
pyo.plot(fig)
