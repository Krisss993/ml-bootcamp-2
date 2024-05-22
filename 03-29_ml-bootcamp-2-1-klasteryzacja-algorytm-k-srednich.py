
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


#############

from numpy.linalg import norm
from sklearn.datasets import make_blobs
import random
from sklearn.cluster import KMeans


#########################
np.random.seed(41)




def odl_euk(x1, x2):
    return (        (  x1[0] - x2[0]  )**2 + (   x1[1] - x2[1]   )**2         )**(1/2)

odl_euk((-1,2), (3,1))





data = make_blobs(n_samples=40, centers=2, cluster_std=1.0, center_box=(-4.0, 4.0), random_state=42)[0]
df = pd.DataFrame(data, columns=['x1', 'x2'])
df.head()


plt.scatter(df['x1'], df['x2'])
plt.show()

fig = px.scatter(df, 'x1', 'x2', width=950, height=500, title='Algorytm K-średnich')
fig.update_traces(marker_size=12)
pyo.plot(fig)





# wyznaczenie wartości brzegowych 
x1_min = df.x1.min()
x1_max = df.x1.max()

x2_min = df.x2.min()
x2_max = df.x2.max()

print(x1_min, x1_max)
print(x2_min, x2_max)


# losowe wygnererowanie współrzędnych centroidów
centroid_1 = np.array([random.uniform(x1_min, x1_max), random.uniform(x2_min, x2_max)])
centroid_2 = np.array([random.uniform(x1_min, x1_max), random.uniform(x2_min, x2_max)])
print(centroid_1)
print(centroid_2)

rand_idx = np.random.choice(len(df), size = 2, replace=False)
points = data[rand_idx,:]


# wizualizacja tzw. punktów startowych centroidów
fig = px.scatter(df, 'x1', 'x2', width=950, height=500, title='Algorytm K-średnich - inicjalizacja centroidów')
fig.add_trace(go.Scatter(x=[centroid_1[0]], y=[centroid_1[1]], name='centroid 1', mode='markers', marker_line_width=3))
fig.add_trace(go.Scatter(x=[centroid_2[0]], y=[centroid_2[1]], name='centroid 2', mode='markers', marker_line_width=3))
fig.update_traces(marker_size=12, showlegend=False)
pyo.plot(fig)

plt.scatter(df['x1'], df['x2'])
plt.scatter(centroid_1[0], centroid_1[1], color='r')
plt.scatter(centroid_2[0], centroid_2[1], color='g')
plt.show()

plt.scatter(df['x1'], df['x2'])
plt.scatter(points[0,0], points[0,1], color='r')
plt.scatter(points[1,0], points[1,1], color='g')
plt.show()





# przypisanie punktów do najbliższego centroidu
clusters = []
for point in data:
    centroid_1_dist = norm(centroid_1 - point)
    centroid_2_dist = norm(centroid_2 - point)
    cluster = 1
    if centroid_1_dist > centroid_2_dist:
        cluster = 2
    clusters.append(cluster)
    
df['cluster'] = clusters
df.head()


# wizualizacja klastrow
fig = px.scatter(df, 'x1', 'x2', width=950, height=500, title='Algorytm K-średnich - inicjalizacja centroidów', color='cluster')
fig.add_trace(go.Scatter(x=[centroid_1[0]], y=[centroid_1[1]], name='centroid 1', mode='markers', marker_line_width=3))
fig.add_trace(go.Scatter(x=[centroid_2[0]], y=[centroid_2[1]], name='centroid 2', mode='markers', marker_line_width=3))
fig.update_traces(marker_size=12, showlegend=False)
pyo.plot(fig)


# obliczenie nowych współrzędnych centroidów
new_centroid_1 = [df[df.cluster == 1].x1.mean(), df[df.cluster == 1].x2.mean()]
new_centroid_2 = [df[df.cluster == 2].x1.mean(), df[df.cluster == 2].x2.mean()]

print(new_centroid_1, new_centroid_2)



# wizualizacja aktualizacji centroidów
fig = px.scatter(df, 'x1', 'x2', color='cluster', width=950, height=500, 
                 title='Algorytm K-średnich - obliczenie nowych centroidów')
#fig.add_trace(go.Scatter(x=[centroid_1[0]], y=[centroid_1[1]], name='centroid 1', mode='markers', marker_line_width=3))
#fig.add_trace(go.Scatter(x=[centroid_2[0]], y=[centroid_2[1]], name='centroid 2', mode='markers', marker_line_width=3))
fig.add_trace(go.Scatter(x=[new_centroid_1[0]], y=[new_centroid_1[1]], name='centroid 1', mode='markers', marker_line_width=3))
fig.add_trace(go.Scatter(x=[new_centroid_2[0]], y=[new_centroid_2[1]], name='centroid 2', mode='markers', marker_line_width=3))
fig.update_traces(marker_size=12)
fig.update_layout(showlegend=False)
pyo.plot(fig)



# obliczenie nowych współrzędnych centroidów
new_centroid_1 = [df[df.cluster == 1].x1.mean(), df[df.cluster == 1].x2.mean()]
new_centroid_2 = [df[df.cluster == 2].x1.mean(), df[df.cluster == 2].x2.mean()]

print(new_centroid_1, new_centroid_2)



# wizualizacja aktualizacji centroidów
fig = px.scatter(df, 'x1', 'x2', color='cluster', width=950, height=500, 
                 title='Algorytm K-średnich - obliczenie nowych centroidów')
#fig.add_trace(go.Scatter(x=[centroid_1[0]], y=[centroid_1[1]], name='centroid 1', mode='markers', marker_line_width=3))
#fig.add_trace(go.Scatter(x=[centroid_2[0]], y=[centroid_2[1]], name='centroid 2', mode='markers', marker_line_width=3))
fig.add_trace(go.Scatter(x=[new_centroid_1[0]], y=[new_centroid_1[1]], name='centroid 1', mode='markers', marker_line_width=3))
fig.add_trace(go.Scatter(x=[new_centroid_2[0]], y=[new_centroid_2[1]], name='centroid 2', mode='markers', marker_line_width=3))
fig.update_traces(marker_size=12)
fig.update_layout(showlegend=False)
pyo.plot(fig)





data = make_blobs(n_samples=40, centers=2, cluster_std=1.0, center_box=(-4.0, 4.0), random_state=42)[0]
df = pd.DataFrame(data, columns=['x1', 'x2'])
df.head()

x1_min = df.x1.min()
x1_max = df.x1.max()

x2_min = df.x2.min()
x2_max = df.x2.max()

centroid_1 = np.array([random.uniform(x1_min, x1_max), random.uniform(x2_min, x2_max)])
centroid_2 = np.array([random.uniform(x1_min, x1_max), random.uniform(x2_min, x2_max)])

for i in range(10):
    clusters = []
    for point in data:
        centroid_1_dist = norm(centroid_1 - point)
        centroid_2_dist = norm(centroid_2 - point)
        cluster = 1
        if centroid_1_dist > centroid_2_dist:
            cluster = 2
        clusters.append(cluster)

    df['cluster'] = clusters

    centroid_1 = [df[df.cluster == 1].x1.mean(), df[df.cluster == 1].x2.mean()]
    centroid_2 = [df[df.cluster == 2].x1.mean(), df[df.cluster == 2].x2.mean()]

print(new_centroid_1, new_centroid_2)



fig = px.scatter(df, 'x1', 'x2', color='cluster', width=950, height=500, 
                 title='Algorytm K-średnich - końcowy rezultat')
fig.add_trace(go.Scatter(x=[new_centroid_1[0]], y=[new_centroid_1[1]], name='centroid 1', mode='markers', marker_line_width=3))
fig.add_trace(go.Scatter(x=[new_centroid_2[0]], y=[new_centroid_2[1]], name='centroid 2', mode='markers', marker_line_width=3))
fig.update_traces(marker_size=12)
fig.update_layout(showlegend=False)
pyo.plot(fig)















data = make_blobs(n_samples=40, centers=2, cluster_std=1.0, center_box=(-4.0, 4.0), random_state=42)[0]
df = pd.DataFrame(data, columns=['x1', 'x2'])
df







class K_mean:
    def __init__(self, max_iters = 5):
        self.max_iters = max_iters
        
    
    
    def fit(self, x, y, display=True):
        data = np.c_[x,y]
        rand_idx = np.random.choice(len(data), 2, replace=False)
        centroid_1 = data[rand_idx[0]]
        centroid_2 = data[rand_idx[1]]
        df = pd.DataFrame(data={'x':x,'y':y})
        
        #for i in range(len(df)):
            #df.loc[i,'Cluser'] = 1 if norm(centroid_1 - (df.loc[i,'x'],df.loc[i,'y'])) < norm(centroid_2 - (df.loc[i,'x'],df.loc[i,'y'])) else 2
    
        df['Cluster2'] = df.apply(lambda z: 1 if norm((centroid_1[0] - z['x'], centroid_1[1] - z['y'])) < norm((centroid_2[0] - z['x'], centroid_2[1] - z['y'])) else 2, axis=1)
        cl1 = df[df['Cluster2'] == 1]
        cl2 = df[df['Cluster2'] == 2]

        
        for i in range(self.max_iters):
            new_centroid_1 = [df[df.Cluster2 == 1].x.mean(), df[df.Cluster2 == 1].y.mean()]
            new_centroid_2 = [df[df.Cluster2 == 2].x.mean(), df[df.Cluster2 == 2].y.mean()]
            df['Cluster2'] = df.apply(lambda z: 1 if norm((new_centroid_1[0] - z['x'], new_centroid_1[1] - z['y'])) < norm((new_centroid_2[0] - z['x'], new_centroid_2[1] - z['y'])) else 2, axis=1)
        
            cl1 = df[df['Cluster2'] == 1]
            cl2 = df[df['Cluster2'] == 2]
            if display:
                plt.scatter(cl1['x'],cl1['y'], color='yellow')
                plt.scatter(cl2['x'],cl2['y'], color='blue')
                plt.scatter(new_centroid_1[0],new_centroid_1[1], color='r')
                plt.scatter(new_centroid_2[0],new_centroid_2[1], color='g')
                plt.title(i)
                plt.show()










k = K_mean()
k.fit(df['x1'], df['x2'])









########################################################################################################################

                                           #   ALGORYTM Z BIBLIOTEKI SKLEARN   #

########################################################################################################################











data = make_blobs(n_samples=1000, centers=None, cluster_std=1.0, center_box=(-4.0, 4.0), random_state=42)[0]
df = pd.DataFrame(data, columns=['x1', 'x2'])
df.head()


fig = px.scatter(df, 'x1', 'x2', width=950, height=500, title='Klasteryzacja - Algorytm K-średnich')
pyo.plot(fig)

kmeans = KMeans(n_clusters=3)
kmeans.fit(data)


y_kmeans = kmeans.predict(data)
y_kmeans[:10]


df['y_kmeans'] = y_kmeans
df




fig = px.scatter(df, 'x1', 'x2', 'y_kmeans', width=950, height=500, title='Algorytm K-średnich - 3 klastry')
pyo.plot(fig)





# WYBÓR OPTYMALNEJ ILOSCI KLASTROW




data = make_blobs(n_samples=1000, centers=4, cluster_std=1.5, center_box=(-8.0, 8.0), random_state=42)[0]
df = pd.DataFrame(data, columns=['x1', 'x2'])

fig = px.scatter(df, 'x1', 'x2', width=950, height=500, title='Algorytm K-średnich', template='plotly_dark')
pyo.plot(fig)


kmeans = KMeans(n_clusters=5)
kmeans.fit(data)

y_kmeans = kmeans.predict(data)

df['y_kmenas'] = y_kmeans
df

fig = px.scatter(df, 'x1', 'x2', color='y_kmenas', width=950, height=500, title='Algorytm K-średnich - 3 klastry')
pyo.plot(fig)


# wcss
kmeans.inertia_


wcss = []
for i in range(2, 10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
    
print(wcss)

plt.plot(range(2,10),wcss)
plt.show()



wcss = pd.DataFrame(wcss, columns=['wcss'])
wcss = wcss.reset_index()
wcss = wcss.rename(columns={'index': 'clusters'})
wcss['clusters'] += 1
wcss.head()

fig = px.line(wcss, x='clusters', y='wcss', width=950, height=500, title='Within-Cluster-Sum of Squared Errors (WCSS)',
        template='plotly_dark')
pyo.plot(fig)



kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

y_kmeans = kmeans.predict(data)

df['y_kmenas'] = y_kmeans
df

fig = px.scatter(df, 'x1', 'x2', color='y_kmenas', width=950, height=500, title='Algorytm K-średnich - 3 klastry')
pyo.plot(fig)


centers = pd.DataFrame(data=kmeans.cluster_centers_, columns=['c1', 'c2'])
centers 

fig = px.scatter(df, 'x1', 'x2', color='y_kmenas', width=950, height=500, title='Algorytm K-średnich - 3 klastry')
fig.add_trace(go.Scatter(x=centers['c1'], y=centers['c2'], mode='markers', 
                         marker={'size': 12, 'color': 'LightSkyBlue', 'line': {'width': 2, 'color': 'tomato'}}, 
                         showlegend=False))
pyo.plot(fig)



plot_decision_regions(data, y_kmeans, kmeans, legend=1)
