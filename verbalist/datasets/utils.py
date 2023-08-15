import matplotlib.pyplot as plt
import numpy as np


def visualize_hist(x: np.ndarray, title: str):
    fig, ax = plt.subplots()
    ax.hist(x, linewidth=0.5, edgecolor="white", bins=300)
    plt.gca().set(title=title, ylabel="Frequency")
    plt.show()
