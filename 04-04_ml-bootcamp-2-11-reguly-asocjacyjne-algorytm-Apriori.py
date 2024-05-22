

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
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
#########################


data = {'produkty': ['chleb jajka mleko', 'mleko ser', 'chleb masło ser', 'chleb jajka']}

transactions = pd.DataFrame(data=data, index=[1, 2, 3, 4])
transactions


# rozwinięcie kolumny do obiektu DataFrame
expand = transactions['produkty'].str.split(expand=True)
expand

product_list = []
for i in range(expand.shape[0]):
    for j in range(expand.shape[1]):
        if expand.iloc[i,j] not in product_list and expand.iloc[i,j]:
            product_list.append(expand.iloc[i,j])
product_list


transactions_encoded = np.zeros((len(expand), len(product_list)))
transactions_encoded

for idx, product in enumerate(product_list):
    for i in range(expand.shape[0]):
        for j in range(expand.shape[1]):
            if expand.iloc[i,j] == product:
                transactions_encoded[i,idx] = 1
transactions_encoded
product_list
expand

transactions_encoded_df = pd.DataFrame(transactions_encoded, columns=product_list)
transactions_encoded_df = transactions_encoded_df.astype('int8')

# % WYSTEPOWANIA PRODUKTOW W OGOLE TRANSAKCJI
supports = apriori(transactions_encoded_df, min_support=0.0000001, use_colnames=True)
supports

supports = apriori(transactions_encoded_df, min_support=0.3, use_colnames=True)
supports

rules = association_rules(supports, metric='confidence', min_threshold=0.65)
rules = rules.iloc[:, [0, 1, 4, 5, 6]]
rules


















pd.set_option('display.float_format', lambda x: f'{x:.2f}')

products = pd.read_csv('https://storage.googleapis.com/esmartdata-courses-files/ml-course/products.csv', usecols=['product_id', 'product_name'])
products.head()

orders = pd.read_csv('https://storage.googleapis.com/esmartdata-courses-files/ml-course/orders.csv', usecols=['order_id', 'product_id'])
orders.head()

data = pd.merge(products, orders, how='inner', on='product_id', sort=True)
data

data = data.sort_values(by='order_id')
data

data.describe()
data.order_id.value_counts()
data.product_name.value_counts()
data['order_id'].nunique()

transactions = data.groupby(by='order_id')['product_name'].apply(lambda name:','.join(name))
transactions

transactions = transactions.str.split(',')
transactions.shape



encoder = TransactionEncoder()
encoder.fit(transactions)
transactions_encoded = encoder.transform(transactions, sparse=True)
transactions_encoded


transactions_encoded_df = pd.DataFrame(transactions_encoded.toarray(), columns=encoder.columns_)
transactions_encoded_df




support = apriori(transactions_encoded_df, min_support=0.01, use_colnames=True)
support = support.sort_values(by='support', ascending=False)
support


rules = association_rules(support, metric='confidence', min_threshold=0)
rules = rules.iloc[:, [0, 1, 4, 5, 6]]
rules = rules.sort_values(by='lift', ascending=False)
rules.head(15)

