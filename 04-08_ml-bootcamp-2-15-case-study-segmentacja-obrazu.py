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
#######################



img = cv2.imread('ski.jpg')
plt.imshow(img)
plt.axis('off')

img.shape
img


# przygotowanie obrazu do modelu
img_data = img.reshape((-1, 3))
img_data = np.float32(img_data)
img_data.shape

df = pd.DataFrame(data=img_data, columns=['dim1', 'dim2', 'dim3'])
df.head(3)

# cv2.kmeans?

_, label, center = cv2.kmeans(
    data=img_data,  # float32 data type
    K=2,            # liczba klastrów
    bestLabels=None,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),  # kryterium zatrzymania (typ, max_iter, eps)
    attempts=10,    # liczba uruchomień algorytmu 
    flags=cv2.KMEANS_RANDOM_CENTERS)    # określenie inicjalizacji centroidów

center = np.uint8(center)
res = center[label.flatten()]
res = res.reshape((img.shape))
plt.imshow(res)
plt.axis('off')


def make_kmeans(n_neighbor=2, img_name='ski.jpg'):

    # wczytanie zdjęcia
    img = cv2.imread(img_name)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # przygotowanie zdjęcia
    img_data = img.reshape((-1, 3))
    img_data = np.float32(img_data)

    # kmeans
    _, label, center = cv2.kmeans(
        data=img_data, 
        K=2, 
        bestLabels=None,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 
        attempts=10, 
        flags=cv2.KMEANS_RANDOM_CENTERS)

    # przygotowanie do wyświetlenia
    center = np.uint8(center)
    res = center[label.flatten()]
    res = res.reshape((img.shape))
    plt.imshow(res)
    plt.axis('off')
    plt.show()
    
make_kmeans(img_name='view.jpg')


