# 9th code



import numpy as np
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot

def local_regression(x0, x, y, tau):
    x0 = np.r_[1, x0]
    x = np.c_[np.ones(len(x)), x]
    xw = x.T * radial_kernal(x0, x, tau)
    beta = np.linalg.pinv(xw @ x) @ xw @ y
    return x @ beta

def radial_kernal(x0, x, tau):
    return np.exp(np.sum((x - x0) ** 2, axis = 1) / (-2 * tau * tau))

def plot_lwr(tau):
    prediction = [local_regression(x0, x, y, tau) for x0 in domain]
    plot = figure(width = 400, height = 400)
    plot.title.text = "tau %g" %tau
    plot.scatter(x, y, alpha = 0.3)
    plot.line(domain, prediction, line_width = 2, color = "red")
    return plot

n = 1000
x = np.linspace(-3, 3, num = n)
y = np.log(np.abs(x ** 2 - 1) + 0.5)

x += np.random.normal(scale = 0.1, size = n)
domain = np.linspace(-3, 3, num = 300)

show(gridplot([[plot_lwr(10.), plot_lwr(1.)],[plot_lwr(0.1), plot_lwr(0.001)]]))



# 8th code

'''

from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

'''

# 7th code

'''

from matplotlib import pyplot as plt
import sklearn.metrics as sm
import numpy as np
import pandas as pd
from sklearn import datasets, preprocessing
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

iris = datasets.load_iris()

x = pd.DataFrame(iris.data)
x.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']

y = pd.DataFrame(iris.target)
y.columns = ['Targets']

model = KMeans(n_clusters = 3, n_init = 10)
model.fit(x)
model.labels_

colormap = np.array(["red", "lime", "black"])
predY = np.choose(model.labels_, [0, 1, 2])
scalar = preprocessing.StandardScaler()
scalar.fit(x)
xsa = scalar.transform(x)
xs = pd.DataFrame(xsa, columns = x.columns)
xs.sample(5)
gmm = GaussianMixture(n_components = 3)
gmm.fit(xs)
y_cluster_gmm = gmm.predict(xs)

plt.figure(figsize = (14, 7))
plt.subplot(1, 2, 1)
plt.scatter(x.Petal_Length, x.Petal_Width, c = colormap[y.Targets], s = 40)
plt.title("Real Classification")

plt.subplot(1, 2, 2)
plt.scatter(x.Petal_Length, x.Petal_Width, c = colormap[model.labels_], s = 40)
plt.title("K Means Classification")

plt.figure(figsize = (14, 7))
plt.subplot(1, 2, 1)
plt.scatter(x.Petal_Length, x.Petal_Width, c = colormap[y_cluster_gmm], s = 40)
plt.title("GMM Classification")

plt.subplot(1, 2, 2)
plt.scatter(x.Petal_Length, x.Petal_Width, c = colormap[predY], s = 40)
plt.title("K Means Classification")

print("K Means accuracy : ", sm.accuracy_score(y, model.labels_))
print("EM accuracy : ", sm.accuracy_score(y, y_cluster_gmm))

plt.show()

'''

# 1st code

'''

def aStarAlgo(start_node, stop_node):
    open_set = set(start_node)
    closed_set = set()
    g = {}
    parents = {}
    g[start_node] = 0
    parents[start_node] = start_node

    while len(open_set) > 0:
        n = None
        for v in open_set:
            if n == None or g[v] + heuristic(v) < g[n] + heuristic(n):
                n = v
        
        if n == stop_node or Graph_nodes[n] == None:
            pass
        else:
            for (m, weight) in get_neighbours(n):
                if m not in open_set and m not in closed_set:
                    open_set.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n
                        if m in closed_set:
                            closed_set.remove(m)
                            open_set.add(m)
        
        if n == None:
            print("No path")
            return None
        if n == stop_node:
            path = []
            while parents[n] != n:
                path.append(n)
                n = parents[n]
            path.append(start_node)
            path.reverse()
            print("Found {}".format(path))
            return path
        open_set.remove(n)
        closed_set.add(n)
    print("No path")
    return None

def heuristic(n):
    H_dist = {
        'A' : 11,
        'B' : 6,
        'C' : 99,
        'D' : 1,
        'E' : 7,
        'G' : 0,
    }
    return H_dist[n]

def get_neighbours(v):
    if v in Graph_nodes:
        return Graph_nodes[v]
    else:
        return None
    
Graph_nodes = {
    'A' : [('B', 2), ('E', 3)],
    'B' : [('C', 1), ('G', 9)],
    'C' : None,
    'E' : [('D', 6)],
    'D' : [('G', 1)]
}

aStarAlgo('A', 'G')

'''

# 5th code

'''

import numpy as np 
x = np.array(([2, 9], [1, 5], [3, 6]))
y = np.array(([92], [86], [89]))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivatives_sigmoid(x):
    return x * (1 - x)

epoch = 1000
lr = 0.1
input_layer = 2
hidden_layer = 3
output_layer = 1

wh = np.random.uniform(size = (input_layer, hidden_layer))
bias_hidden = np.random.uniform(size = (1, hidden_layer))
weight_hidden = np.random.uniform(size = (hidden_layer, output_layer))
bias_output = np.random.uniform(size = (1, output_layer))

for i in range(epoch):
    hinp1 = np.dot(x, wh)
    hinp = hinp1 + bias_hidden
    hlayer_activation = sigmoid(hinp)

    outinp1 = np.dot(hlayer_activation, weight_hidden)
    outinp = outinp1 + bias_output
    output = sigmoid(outinp)

    EO = y - output

    outgrad = derivatives_sigmoid(output)
    d_output = EO * outgrad

    EH = d_output.dot(weight_hidden.T)
    hiddengrad = derivatives_sigmoid(hlayer_activation)
    d_hiddenlayer = EH * hiddengrad

    weight_hidden += hlayer_activation.dot(d_output) * lr
    bias_hidden += np.sum(d_hiddenlayer, axis = 0, keepdims = True) * lr
    wh += x.T.dot(d_hiddenlayer) * lr
    bias_output += np.sum(d_output, axis = 0, keepdims = True) * lr

print("Input : \n" + str(x))
print("Actual output : \n" + str(y))
print("Predicted output : \n" + str(output))

'''

# 3rd code

'''

import numpy as np
import pandas as pd

data = pd.DataFrame(data=pd.read_csv("E:\\Python\\DataSets\\_03_Third.csv"))

concepts = np.array(data.iloc[ : , 0 : -1])
print(concepts)

target = np.array(data.iloc[ : , -1])
print(target)

def learn (concepts, target):
    specific_h = concepts[0].copy()
    print("Initialization of specific_h and general_h")
    print(specific_h)

    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
    print(general_h)

    for i, h in enumerate(concepts):
        print("For loop starts")
        if target[i] == "yes":
            print("If instance is positive")
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = "?"
                    general_h[x][x] = "?"
        
        if target[i] == "no":
            print("If instance is Negative")
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = "?"
        
        print("Steps of candidate elimination algorithm", i + 1)
        print(specific_h)
        print(general_h)
        print("\n")
        print("\n")

    indices = [i for i, val in enumerate(general_h) if val == ["?", "?", "?", "?", "?", "?"]]

    for i in indices:
        general_h.remove(["?", "?", "?", "?", "?", "?"])
    return specific_h, general_h

s_final, g_final = learn(concepts, target)

print("Final Specific_h:", s_final, sep = "\n")
print("Final General_h:", g_final, sep = "\n")

'''