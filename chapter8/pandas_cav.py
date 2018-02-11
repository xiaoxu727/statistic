import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import matplotlib.pyplot as plt
def line():
    fig = plt.Figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    s = Series(np.random.randn(10).cumsum(), index=np.arange(0, 100, 10))
    s.plot()
    plt.show()
    # print(s)
def multi_lines():

    df = DataFrame(np.random.randn(10, 4).cumsum(0),
                   columns=['A', 'B', 'C', 'D'],
                   index=np.arange(0, 100, 10))
    plt.show()
def bar():
    fig, axes = plt.subplots(2, 1)
    data = Series(np.random.rand(16), index=list('abcdefghijklmnop'))
    data.plot(kind='bar', ax=axes[1], color='k', alpha=0.7)
    data.plot(kind='barh', ax=axes[0], color='k', alpha=0.7)
    plt.show()
if __name__ == '__main__':
    # line()
    # multi_lines()
    bar()