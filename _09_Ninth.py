import numpy as np
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot

def local_regression(x0, x, y, tau):
    x0 = np.r_[1, x0]
    x = np.c_[np.ones(len(x)), x]
    xw = x.T * radial_kernel(x0, x, tau)
    beta = np.linalg.pinv(xw @ x) @ xw @ y
    return x @ beta

def radial_kernel(x0, x, tau):
    return np.exp(np.sum((x - x0) ** 2, axis=1) / (-2 * tau * tau))

n = 1000
x = np.linspace(-3, 3, num=n)
print("The data set (10 samples) x:\n", x[1:10])
y = np.log(np.abs(x ** 2 - 1) + 0.5)
print("The fitting curve data set (10 samples) y:\n", y[1:10])

x += np.random.normal(scale=0.1, size=n)
print("Normalized (10 samples) x:\n", x[1:10])

domain = np.linspace(-3, 3, num=300)
print("x0 domain space (10 samples):\n", domain[1:10])

def plot_lwr(tau):
    prediction = [local_regression(x0, x, y, tau) for x0 in domain]
    plot = figure(width=400, height=400)
    plot.title.text = "tau = %g" % tau
    plot.scatter(x, y, alpha=0.3)
    plot.line(domain, prediction, line_width=2, color="red")
    return plot

show(gridplot([[plot_lwr(10.), plot_lwr(1.)], [plot_lwr(0.1), plot_lwr(0.01)]]))
