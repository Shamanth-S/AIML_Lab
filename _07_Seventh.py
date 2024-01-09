import pandas as pd
import numpy as np
from sklearn import datasets
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture

iris = datasets.load_iris()

x = pd.DataFrame(iris.data)
x.columns = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]

y = pd.DataFrame(iris.target)
y.columns = ["Targets"]

model = KMeans(n_clusters = 3, n_init = 10)
model.fit(x)
model.labels_

plt.figure(figsize = (14, 7))
colormap = np.array(["red", "lime", "black"])
plt.subplot(1, 2, 1)
plt.scatter(x.Petal_Length, x.Petal_Width, c = colormap[model.labels_], s = 40)
plt.title("K Mean Classification")

plt.figure(figsize = (14, 7))
predY = np.choose(model.labels_, [0, 1, 2])
plt.subplot(1, 2, 1)
plt.scatter(x.Petal_Length, x.Petal_Width, c = colormap[y.Targets], s = 40)
plt.title("Real Classification")

plt.subplot(1, 2, 2)
plt.scatter(x.Petal_Length, x.Petal_Width, c = colormap[predY], s = 40)
plt.title("K Mean Classification")

print("The accuracy score of K-Mean : \n", sm.accuracy_score(y, model.labels_))
print("The confusion matrix of K-Means : \n", sm.confusion_matrix(y, model.labels_))


scaler = preprocessing.StandardScaler()
scaler.fit(x)
xsa = scaler.transform(x)
xs = pd.DataFrame(xsa, columns = x.columns)
xs.sample(5)

gmm = GaussianMixture(n_components = 3)
gmm.fit(xs)

y_cluster_gmm = gmm.predict(xs)
plt.subplot(2, 2, 3)
plt.scatter(x.Petal_Length, x.Petal_Width, c = colormap[y_cluster_gmm], s = 40)
plt.title("GMM classification")

print("The accuracy score of EM : \n", sm.accuracy_score(y, y_cluster_gmm))
print("The confusion matrix of EM : \n", sm.confusion_matrix(y, y_cluster_gmm))

plt.show()
