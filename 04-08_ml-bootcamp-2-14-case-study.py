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
from sklearn.cluster import KMeans, DBSCAN


###################
import prophet
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.preprocessing import StandardScaler
#######################
prophet.__version__

url = 'https://storage.googleapis.com/esmartdata-courses-files/ml-course/OnlineRetail.csv'
raw_data = pd.read_csv(url, encoding='latin', parse_dates=['InvoiceDate'])
data = raw_data.copy()
data.head(3)

data.info()
data.describe()
data.describe(include=['object'])
data.describe(include=['datetime'])


# usunięcie braków
data.isnull().sum()

data = data.dropna()
data.isnull().sum()

data['Country'].value_counts()


tmp = data['Country'].value_counts().nlargest(10).reset_index()


# google -> color picker
fig = px.bar(tmp, x='Country', y='count', template='plotly_dark', color_discrete_sequence=['#03fcb5'],
       title='Częstotliwość zakupów ze względu na kraj')
pyo.plot(fig)

# obcięcie tylko do United Kingdom
data_uk = data[data['Country'] == 'United Kingdom'].copy()
data_uk = data.query("Country == 'United Kingdom'").copy()

# utworzenie nowej zmiennej Sales
data_uk['Sales'] = data_uk['Quantity'] * data_uk['UnitPrice']



# częstotliwość zakupów ze względu na datę
tmp = data_uk.groupby(data_uk['InvoiceDate'].dt.date)['CustomerID'].count().reset_index()
tmp.columns = ['InvoiceDate', 'Count']
tmp.head()

from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

trace1 = px.line(tmp, x='InvoiceDate', y='Count', template='plotly_dark', color_discrete_sequence=['#03fcb5'])['data'][0]
trace2 = px.scatter(tmp, x='InvoiceDate', y='Count', template='plotly_dark', color_discrete_sequence=['#03fcb5'])['data'][0]

fig.add_trace(trace1, row=1, col=1)
fig.add_trace(trace2, row=2, col=1)
fig.update_layout(template='plotly_dark', title='Częstotliwość zakupów ze względu na datę', width=950)
pyo.plot(fig)

data_uk.columns


# Łączna sprzedaż ze względu na datę
tmp = data_uk.groupby(data_uk['InvoiceDate'].dt.date)['Sales'].sum().reset_index()
tmp.columns = ['InvoiceDate', 'Sales']
tmp.head()


fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

trace1 = px.line(tmp, x='InvoiceDate', y='Sales', template='plotly_dark', color_discrete_sequence=['#03fcb5'])['data'][0]
trace2 = px.scatter(tmp, x='InvoiceDate', y='Sales', template='plotly_dark', color_discrete_sequence=['#03fcb5'])['data'][0]

fig.add_trace(trace1, row=1, col=1)
fig.add_trace(trace2, row=2, col=1)
fig.update_layout(template='plotly_dark', title='Łączna sprzedaż ze względu na datę', width=950)
pyo.plot(fig)









# # # # # # # #  # # # # # # #                         WYZNACZENIE RETENCJI DLA KAZDEGO KLIENTA                       # # # # # # # # # # # # # # # 














# wydobycie unikalnych wartości CustomerID
data_user = pd.DataFrame(data=data['CustomerID'].unique(), columns=['CustomerID'])
data_user.head(3)



# wydobycie daty ostatniego zakupu dla każdego klienta
last_purchase = data_uk.groupby(by=data['CustomerID'])['InvoiceDate'].max().reset_index()
last_purchase.columns = ['CustomerID', 'LastPurchaseDate']
last_purchase.head()

last_purchase['LastPurchaseDate'].max()
last_purchase['LastPurchaseDate'].min()



# wyznaczenie retencji jako liczby dni od daty ostatniego kupna klienta do maksymalnej (ostatniej) daty kupna w danych
last_purchase_date = data_uk['InvoiceDate'].max()
last_purchase['Retention'] = (last_purchase_date - last_purchase['LastPurchaseDate']).dt.days
last_purchase.head()

last_purchase['Retention'].value_counts()



plt.hist(last_purchase['Retention'], bins=50)

fig = px.histogram(last_purchase, x='Retention', template='plotly_dark', 
             width=950, height=500, title='Retention', nbins=100, 
             color_discrete_sequence=['#03fcb5'])
pyo.plot(fig)


data_user = pd.DataFrame(data=np.c_[last_purchase['Retention'], last_purchase['CustomerID']], columns=['Retention', 'CustomerID'])
data_user.head()


data_retention = data_user[['Retention']]
data_retention.head()



# standaryzacja danych
scaler = StandardScaler()
data_user['RetentionScaled'] = scaler.fit_transform(data_retention)
data_user.head()


fig = px.scatter(data_user, x='CustomerID', y='RetentionScaled', template='plotly_dark', width=950,
           color_discrete_sequence=['#03fcb5'])
pyo.plot(fig)

data_retention_scaled = data_user[['RetentionScaled']]
data_retention_scaled.head()

wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, max_iter=1000)
    kmeans.fit(data_retention_scaled)
    wcss.append(kmeans.inertia_)

wcss = pd.DataFrame(data=np.c_[range(1, 10), wcss], columns=['NumberOfClusters', 'WCSS'])
wcss


fig = px.line(wcss, x='NumberOfClusters', y='WCSS', template='plotly_dark', title='WCSS', 
        width=950, color_discrete_sequence=['#03fcb5'])
pyo.plot(fig)


kmeans = KMeans(n_clusters=3, max_iter=1000)
kmeans.fit(data_retention_scaled)

data_user['Cluster'] = kmeans.labels_
data_user.head()



tmp = data_user.groupby('Cluster')['Retention'].describe()
tmp

tmp = tmp['mean'].reset_index()
tmp.columns = ['Cluster', 'MeanRetention']
fig = px.bar(tmp, x='Cluster', y='MeanRetention', template='plotly_dark', width=950, 
       height=400, color_discrete_sequence=['#03fcb5'])
pyo.plot(fig)

fig = px.scatter(data_user, x='CustomerID', y='Retention', color='Cluster', template='plotly_dark', 
           width=950, title='Wizualizacja klastrów')
pyo.plot(fig)














data_user = pd.DataFrame(data=np.c_[last_purchase['Retention'], last_purchase['CustomerID']], columns=['Retention', 'CustomerID'])
data_user.head()


data_retention = data_user[['Retention']]
data_retention.head()



# standaryzacja danych
scaler = StandardScaler()
data_user['RetentionScaled'] = scaler.fit_transform(data_retention)
data_user.head()


dbscan = DBSCAN(eps=0.03, min_samples=5)
dbscan.fit(data_retention_scaled)
clusters = dbscan.labels_
data_user['Cluster'] = clusters
data_user.head()
     

fig = px.scatter(data_user, x='CustomerID', y='Retention', color='Cluster', template='plotly_dark', width=950,
           title='Wizualizacja klastrów')
pyo.plot(fig)



data_sales = data_uk.groupby('CustomerID')['Sales'].sum().reset_index()
data_sales.head()

data_user = pd.merge(data_user, data_sales, on='CustomerID')
data_user

scaler = StandardScaler()
data_user['SalesScaled'] = scaler.fit_transform(data_user[['Sales']])
data_user.head()

fig = px.scatter(data_user, x='CustomerID', y='Sales', template='plotly_dark',
           color_discrete_sequence=['#03fcb5'], title='Sprzedaż w rozbiciu na klienta')
pyo.plot(fig)

fig = px.scatter(data_user, x='CustomerID', y='SalesScaled', template='plotly_dark',
           color_discrete_sequence=['#03fcb5'], title='Sprzedaż w rozbiciu na klienta - dane przeskalowane')
pyo.plot(fig)


data_sales_scaled = data_user[['SalesScaled']]
data_sales_scaled.head()

wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, max_iter=1000)
    kmeans.fit(data_sales_scaled)
    wcss.append(kmeans.inertia_)

wcss = pd.DataFrame(data=np.c_[range(1, 10), wcss], columns=['NumberOfClusters', 'WCSS'])
wcss

fig = px.line(wcss, x='NumberOfClusters', y='WCSS', template='plotly_dark', color_discrete_sequence=['#03fcb5'], 
        width=950, title='WCSS')
pyo.plot(fig)



kmeans = KMeans(n_clusters=3, max_iter=1000)
kmeans.fit(data_sales_scaled)

data_user['Cluster'] = kmeans.labels_
data_user['Cluster'] = data_user['Cluster'].astype(str)
data_user.head()

kmeans.cluster_centers_

fig = px.scatter(data_user, x='CustomerID', y='SalesScaled', color='Cluster', template='plotly_dark', width=950,
           title='Wizualizacja klastrów - dane przeskalowane')
pyo.plot(fig)






dbscan = DBSCAN(eps=0.5, min_samples=7)
dbscan.fit(data_sales_scaled)
clusters = dbscan.labels_
data_user['Cluster'] = clusters
data_user['Cluster'] = data_user['Cluster'].astype(str)
data_user.head()


fig = px.scatter(data_user, x='CustomerID', y='Sales', color='Cluster', template='plotly_dark', width=950,
           title='DBSCAN - Wizualizacja klastrów')
pyo.plot(fig)


data_scaled = data_user[['RetentionScaled','SalesScaled']]




wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, max_iter=1000)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

wcss = pd.DataFrame(data=np.c_[range(1, 10), wcss], columns=['NumberOfClusters', 'WCSS'])
wcss   
     
fig = px.line(wcss, x='NumberOfClusters', y='WCSS', template='plotly_dark', color_discrete_sequence=['#03fcb5'], width=950,
        title='WCSS')
pyo.plot(fig)


kmeans = KMeans(n_clusters=5, max_iter=1000)
kmeans.fit(data_scaled)


data_user['Cluster'] = kmeans.labels_
data_user['Cluster'] = data_user['Cluster'].astype(str)
data_user.head()


fig = px.scatter(data_user, x='RetentionScaled', y='SalesScaled', color='Cluster', template='plotly_dark', width=950,
           title='KMeans - Wizualizacja klastrów')
pyo.plot(fig)



centroids = kmeans.cluster_centers_
centroids


fig = px.scatter(data_user, x='RetentionScaled', y='SalesScaled', color='Cluster', template='plotly_dark', width=900,
                 title='KMeans - Wizualizacja klastrów + centroidy')
fig.add_trace(go.Scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers', marker_symbol='star',
                         marker_size=10, marker_color='white', showlegend=False))
pyo.plot(fig)


desc = data_user.groupby('Cluster')[['Retention', 'Sales']].describe()
desc



tmp = pd.merge(desc['Retention'][['count', 'mean']].reset_index(), desc['Sales'][['mean']].reset_index(), on='Cluster',
         suffixes=('_Retention', '_Sales'))
tmp
     
fig = px.bar(tmp, x='count', y='Cluster', hover_data=['mean_Retention', 'mean_Sales'], template='plotly_dark', 
       width=950, orientation='h', title='Rozkład klastrów')
pyo.plot(fig)

