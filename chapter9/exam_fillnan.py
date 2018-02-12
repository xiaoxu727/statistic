import numpy as np
import pandas as pd
from pandas import DataFrame, Series


def fillnan1():
    s = Series(np.random.randn(6))
    print(s)
    s[::2] = np.nan
    print(s)
    print(s.fillna(s.mean()))
    # print(s)


def fillnan2():
    states = ['Ohio', 'New York', 'Vermont', 'Florida', 'Oregon', 'Nevada', 'California', 'Idaho']
    group_key = ['East'] * 4 + ['West'] * 4
    data = Series(np.random.randn(8), index=states)
    data[['Vermont', 'Nevada', 'Idaho']] = np.nan
    print('--------------------')
    print(data)
    print('--------------------')
    print(data.groupby(group_key).mean())
    fill_mean = lambda g: g.fillna(g.mean())
    print('--------------------')
    print(data.groupby(group_key).apply(fill_mean))
    print('--------------------')
    fill_values = {'East': 0.5, 'West': -1}
    fill_func = lambda g: g.fillna(fill_values[g.name])
    print(data.groupby(group_key).apply(fill_func))
    print('--------------------')
    fill_func2 = lambda g: print('----',g.name,'|',type(g),'|', g)
    print(data.groupby(group_key).apply(fill_func2))
    # grouped = data.groupby(group_key)
    # for name, value in grouped:
    #     print(name, value)


if __name__ == '__main__':
    # fillnan1()
    fillnan2()