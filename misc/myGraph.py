import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def myGraph(model, X_train = None, y_train = None, labels = [], show_scatter = True):    
    _, ax = plt.subplots(figsize = (10,10))
    ax.set_aspect("equal")

    if model != None:
        minX = min(X_train[:, 0])
        maxX = max(X_train[:, 0])
        minY = min(X_train[:, 1])
        maxY = max(X_train[:, 1])

        marginX = (maxX - minX) * 0.1
        marginY = (maxY - minY) * 0.1

        x = np.linspace(minX - marginX, maxX + marginX, 1000)
        y = np.linspace(minY - marginY, maxY + marginY, 1000)
        X, Y = np.meshgrid(x, y)
        Z = model.predict(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
        ax.contourf(X, Y, Z, levels = 2, colors = ["#0079FF", "#00DFA2", "#FF0060"], zorder = 0)
    if show_scatter:
        scatter = ax.scatter(
            x = X_train[:, 0], y = X_train[:, 1], c = y_train,
            cmap = ListedColormap(["#0079FF", "#00DFA2", "#FF0060"])  , zorder = 2, edgecolor = "#000000"
        )
    
    if (len(labels) > 0) and show_scatter:
        ax.legend(
            handles = scatter.legend_elements()[0],
            labels = list(labels)
        )

    ax.grid(color = "#eeeeee", zorder = 1, alpha = 0.4)
    
    return plt.plot()