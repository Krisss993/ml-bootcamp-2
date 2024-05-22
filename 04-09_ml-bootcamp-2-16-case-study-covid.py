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
import cv2
import prophet
from prophet import Prophet
from prophet.plot import plot_plotly
#######################

# dane od 22.01.2020 do 17.02.2020
url = 'https://storage.googleapis.com/esmartdata-courses-files/ml-course/coronavirus.csv'
data = pd.read_csv(url, parse_dates=['Date', 'Last Update'])
data.head()

data.info()


data.isnull().sum()


# brak Province/State -> Country
data['Province/State'] = np.where(data['Province/State'].isnull(), data['Country'], data['Province/State'])
data.isnull().sum()

data['Country'].value_counts()[:30]

data['Country'] = np.where(data['Country'] == 'Mainland China', 'China', data['Country'])

data['Country'].value_counts().nlargest(10)


tmp = data['Country'].value_counts().nlargest(15).reset_index()
tmp.columns = ['Country', 'Count']
tmp = tmp.sort_values(by=['Count', 'Country'], ascending=[False, True])
tmp['iso_alpha'] = ['CHN', 'USA', 'AUS', 'CAN', 'JPN', 'KOR', 'THA', 'HKG', np.nan, 'SGP', 'TWN', 'VNM', 'FRA', 'MYS', 'NPL'] 
tmp

fig = px.scatter_geo(tmp, locations='iso_alpha', size='Count', size_max=40, template='plotly_dark', color='Count',
               text='Country', projection='natural earth', color_continuous_scale='reds', width=950,
               title='Liczba przypadków Koronawirusa na świcie - TOP15')
pyo.plot(fig)

fig = px.scatter_geo(tmp, locations='iso_alpha', size='Count', size_max = 40, template='plotly_dark', color='Count',
               text='Country', projection='natural earth', color_continuous_scale='reds', scope='asia', width=950,
               title='Liczba przypadków Koronawirusa - Azja (z TOP15 global)')
pyo.plot(fig)

fig = px.bar(tmp, x='Country', y='Count', template='plotly_dark', width=950, color_discrete_sequence=['#42f5c8'],
       title='Liczba przypadków Koronawirusa w rozbiciu na kraje')
pyo.plot(fig)



fig = px.bar(tmp.query("Country != 'China'"), x='Country', y='Count', template='plotly_dark', width=950, 
       color_discrete_sequence=['#42f5c8'], title='Liczba przypadków Koronawirusa w rozbiciu na kraje (poza Chinami)')
pyo.plot(fig)



tmp = data.groupby(by=data['Date'].dt.date)[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
tmp



fig = go.Figure()

trace1 = go.Scatter(x=tmp['Date'], y=tmp['Confirmed'], mode='markers+lines', name='Confirmed')
trace2 = go.Scatter(x=tmp['Date'], y=tmp['Deaths'], mode='markers+lines', name='Deaths')
trace3 = go.Scatter(x=tmp['Date'], y=tmp['Recovered'], mode='markers+lines', name='Recovered')

fig.add_trace(trace1)
fig.add_trace(trace2)
fig.add_trace(trace3)

fig.update_layout(template='plotly_dark', width=950, title='Koronawirus (22.01-17.02.2020)')
pyo.plot(fig)


data_confirmed = tmp[['Date', 'Confirmed']]
data_confirmed.columns = ['ds', 'y']
data_confirmed.head()

fig = go.Figure()
fig.add_trace(go.Scatter(x=data_confirmed['ds'], y=data_confirmed['y'], mode='markers+lines',
                         name='Confirmed', fill='tozeroy'))
fig.update_layout(template='plotly_dark', width=950, title='Liczba potwierdzonych przypadków (22.01-12.02)')
pyo.plot(fig)

# dopasowanie modelu
model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
model.fit(data_confirmed)

# predykcja
future = model.make_future_dataframe(periods=7, freq='D')
forecast = model.predict(future)
fig = plot_plotly(model, forecast)
pyo.plot(fig)
