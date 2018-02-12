# 分组加权平均数和相关系数
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

def test1():
    df = DataFrame({'category': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
                    'data': np.random.randn(8),
                    'weights': np.random.randn(8)})
    print(df)
    grouped = df.groupby('category')
    get_wavg = lambda x: np.average(x['data'], weights=x['weights'])
    print(grouped.apply(get_wavg))

def yahoo_finance():


if __name__ == '__main__':
    test1()
