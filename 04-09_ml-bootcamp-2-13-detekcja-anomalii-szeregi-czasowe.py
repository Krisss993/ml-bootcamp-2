


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


###################
import prophet
from prophet import Prophet
from prophet.plot import plot_plotly
#######################
prophet.__version__

df = pd.read_csv('https://storage.googleapis.com/esmartdata-courses-files/ml-course/traffic.csv', 
                 parse_dates=['timestamp'])
df.head()
df.info()



fig = px.line(df, x='timestamp', y='count', title='Anomaly Detection - web traffic', width=950, height=500,
        template='plotly_dark', color_discrete_sequence=['#42f5d4'])
pyo.plot(fig)

fig = px.scatter(df, x='timestamp', y='count', title='Anomaly Detection - web traffic', width=950, height=500,
           template='plotly_dark', color_discrete_sequence=['#42f5d4'])
pyo.plot(fig)






# Danymi wejściowymi do klasy Prophet jest obiekt DataFrame biblioteki pandas. Wejściowy DataFrame składa się z dwóch kolumn:

# ds (datestamp, odpowiednio sformatowana kolumna, np. YYYY-MM-DD dla daty, YYYY-MM-DD HH:MM:SS dla dokładnego czasu )
# y (kolumna numeryczna, reprezentująca wartość, którą chcemy przewidywać)







data = df.copy()
data.columns = ['ds', 'y']
data.head(3)

#Prophet?

model = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=False, 
                interval_width=0.99, changepoint_range=0.8)

model.fit(data)
forecast = model.predict(data)

forecast[['ds', 'trend', 'yhat', 'yhat_lower', 'yhat_upper']].head(3)

forecast.columns

forecast['real'] = data['y']
forecast['anomaly'] = 1

# PUNKTY POZA WARTOSCIAMI yhat_lower - yhat_upper sa traktowane jako outliery
forecast.loc[forecast['real'] > forecast['yhat_upper'], 'anomaly'] = -1
forecast.loc[forecast['real'] < forecast['yhat_lower'], 'anomaly'] = -1
forecast.head(3)

fig = px.scatter(forecast, x='ds', y='real', color='anomaly', color_continuous_scale='Bluyl', 
           title='Anomaly Detection in Time Series', template='plotly_dark', width=950, height=500)
pyo.plot(fig)










# przewidzenie wartosci na jeden dzien
future = model.make_future_dataframe(periods=1440, freq='Min')
future

forecast = model.predict(future)
forecast[['ds', 'trend', 'yhat', 'yhat_lower', 'yhat_upper']].tail(3)




# CZARNE PNKTY TO DANE TRENINGOWE
# LINIA NIEBIESKA TO PREDYKCJA MODELU 
# JASNONIEBIESKIE POLE TO PRZEDZIAL yhat_lower - yhat_upper
_ = model.plot(forecast)




# WYKRES SEZONOWOSCI ZGODNY Z MODELEM
_ = model.plot_components(forecast)


# WYKRES PLOTLY
fig = plot_plotly(model, forecast, xlabel='czas', ylabel='ruch webowy')
pyo.plot(fig)




data_sep = df[df['timestamp'].dt.month == 9]
data_sep.columns = ['ds','y']
data_sep.head()


model = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=False, 
                interval_width=0.99, changepoint_range=0.8)


model.fit(data_sep)

forecast = model.predict(data_sep)

forecast['real'] = data_sep['y']
forecast['anomaly'] = 1

forecast.loc[forecast['real'] > forecast['yhat_upper'], 'anomaly'] = -1
forecast.loc[forecast['real'] < forecast['yhat_lower'], 'anomaly'] = -1
forecast.head(3)

fig = px.scatter(forecast, x='ds', y='real', color='anomaly', color_continuous_scale='Bluyl', 
           title='Anomaly Detection in Time Series', template='plotly_dark', width=950, height=500)
pyo.plot(fig)


future = model.make_future_dataframe(periods=1440, freq='Min')
future
     

forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()

_ = model.plot(forecast)
     
_ = model.plot_components(forecast)




fig = plot_plotly(model, forecast, xlabel='czas', ylabel='ruch webowy')
pyo.plot(fig)

