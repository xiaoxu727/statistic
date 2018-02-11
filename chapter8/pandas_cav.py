import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import matplotlib.pyplot as plt


def line():
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    s = Series(np.random.randn(10).cumsum(), index=np.arange(0, 100, 10))
    s.plot(ax=ax2)
    plt.show()
    # print(s)


def multi_lines():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    df = DataFrame(np.random.randn(10, 4).cumsum(0),
                   columns=['A', 'B', 'C', 'D'],
                   index=np.arange(0, 100, 10))
    df.plot(ax=ax)
    plt.show()


# 柱状图
def bar():
    fig, axes = plt.subplots(2, 1)
    data = Series(np.random.rand(16), index=list('abcdefghijklmnop'))
    data.plot(kind='bar', ax=axes[1], color='k', alpha=0.7)
    data.plot(kind='barh', ax=axes[0], color='k', alpha=0.7)
    plt.show()
def bar2():
    fig,axes = plt.subplots(2,1)
    df = DataFrame(np.random.rand(6,4),
                   index=['one','two','three','four','five','six'],
                   columns=pd.Index(['A','B','C','D'],name='Genus'))
    print(df)
    df.plot(kind='bar',ax=axes[0])
    df.plot(kind='barh', stacked=True, alpha=0.5, ax=axes[1])

    plt.show()
def bar3():
    tips = pd.read_csv('data/tips.csv')
    party_counts = pd.crosstab(tips.day, tips['size'])
    party_counts = party_counts.ix[:, 2:5]
    print(party_counts)
    party_pcts = party_counts.div(party_counts.sum(1).astype(float),axis=0)
    print(party_pcts)
    party_pcts.plot(kind='bar', stacked=True)
    plt.show()
def histogram():
    tips = pd.read_csv('data/tips.csv')
    tips['tip_pct'] = tips['tip'] / tips['total_bill']
    tips['tip_pct'].hist(bins=50)
    plt.show()
def kde():
    tips = pd.read_csv('data/tips.csv')
    tips['tip_pct'] = tips['tip'] / tips['total_bill']
    tips['tip_pct'].plot(kind='kde')
    plt.show()
def kde_histogram():
    comp1 = np.random.normal(0, 1, size=200)
    comp2 = np.random.normal(10, 2, size=200)
    values = Series(np.concatenate([comp1, comp2]))
    values.hist(bins=100, alpha=0.3, color='k', normed=True)
    values.plot(kind='kde', style='k--')
    plt.show()
# 散布图
def scatter():
    macro = pd.read_csv('data/macrodata.csv')

    data = macro['cpi','m1','tbilrate','unemp']
    trans_data = np.log(data).diff().dropna()
    print(trans_data)
    print(trans_data[-5:])
    plt.scatter(trans_data['m1'], trans_data['unemp'])
    plt.title(' Change in log %s vs . log %s' %('m1', 'unemp'))
    plt.show()
def scatter2():
    macro = pd.read_csv('data/macrodata.csv')
    print(macro)
    data = macro[['cpi', 'm1', 'tbilrate', 'unemp']]

    trans_data = np.log(data).diff().dropna()
    print(data)
    pd.plotting.scatter_matrix(trans_data, diagonal='kde', color='k', alpha=0.3)
    plt.show()
if __name__ == '__main__':
    # line()
    # multi_lines()
    # bar()
    # bar2()
    # bar3()
    # histogram()
    # kde()
    # kde_histogram()
    # scatter()
    scatter2()