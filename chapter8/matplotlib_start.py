from io import StringIO
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from utils.logger.logger import logger
import numpy as np
from numpy.random import randn
def figure():
    logger.info('Figure 和 Subplot')
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax1.hist(randn(100), bins=20, color='k', alpha=0.3)
    ax2.scatter(np.arange(30), np.arange(30) + 3 * randn(30))
    plt.plot(randn(50).cumsum(), 'k--')
    fig.show()
def subplot():
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=False)
    for i in range(2):
        for j in range(2):
            axes[i, j].hist(randn(500), bins=50, color='k', alpha=0.5)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

    # time.sleep(100)
# 颜色、标记和线性
def color_label_line():
    a = randn(30)
    print(a)
    a = a.cumsum()
    print(a)
    plt.legend(loc='best')
    plt.plot(a, 'ko--')
    # 等价于
    # plt.plot(a,color='k', marker='o', linestyle='dashed')
    plt.show()
# 刻度、标签和图例
def ticks_label_legend():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(randn(1000).cumsum(), 'k', label='one') # label 用来标注legend
    ax.plot(randn(1000).cumsum(), 'k--', label='two')
    ax.plot(randn(1000).cumsum(), 'k--', label='three')
    ticks = ax.set_xticks([0, 250, 500, 750, 1000])
    lables = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'],
                                rotation=30, fontsize='small')
    ax.set_title('My first matplotlib plot')
    ax.set_xlabel('Stages')
    ax.legend(loc='best')
    plt.show()
def note():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    data = pd.read_csv('data/spx.csv',index_col=0, parse_dates=True)
    spx = data['SPX']
    spx.plot(ax=ax, style='k-')
    crisis_data=[
        (datetime(2007, 10, 11),'Peak of bull market'),
        (datetime(2008, 3, 12),'Bear Stearns Fails'),
        (datetime(2008, 9, 15),'Lehman Bankruptcy')
    ]
    for date, label in crisis_data:
        ax.annotate(label, xy=(date, spx.asof(date)+50),
                        xytext=(date, spx.asof(date)+200),
                    arrowprops=dict(facecolor='black'),
                    horizontalalignment='left', verticalalignment='top')
#         放大到2007-2010
    ax.set_xlim(['1/1/2007', '1/1/2011'])
    ax.set_ylim([600, 1800])
    ax.set_title('Important dates in 2008-2009 financial crisis')
    plt.show()
#     绘制图形 and save
def patch():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    rect = plt.Rectangle((0.2,0.75), 0.4, 0.15, color='k', alpha=0.3)
    circ = plt.Circle((0.7,0.2), 0.15, color='b', alpha=0.3)
    pgon = plt.Polygon([[0.15,0.15], [0.35, 0.4], [0.2, 0.6]], color='g', alpha=0.5)
    ax.add_patch(rect)
    ax.add_patch(circ)
    ax.add_patch(pgon)
    plt.savefig('patch.jpg')
    plt.savefig('patch.pdf')
    plt.savefig('patch.png', dpi=400, bbox_inches='tight')
    # 保存IO不支持
    # buffer = StringIO()
    # plt.savefig(buffer)
    # plot_data = buffer.getvalue()
    # print(plot_data)
    plt.show()
def matplotlib_config():
    plt.rc('figure',figsize=(8, 8))
    font_options = {
        'family': 'monospace',
        'weight': 'bold',
        'size': 40
    }
    plt.rc('font', **font_options)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(randn(100), bins=10)
    plt.show()

# save table to file
# def save_table():


if __name__ == '__main__':
    # figure()
    # subplot()
    # color_label_line()
    # ticks_label_legend()
    # note()
    # patch()
    matplotlib_config()