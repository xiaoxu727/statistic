import pandas as pd
import numpy as np
from pandas import DataFrame, Series

def groupby1():
    df = DataFrame({'key1':['a', 'a', 'b', 'b', 'a'],
                    'key2':['one', 'two', 'three', 'four', 'five'],
                    'data1': np.random.randn(5),
                    'data2': np.random.randn(5)})
    print(df)
    # 方法一
    grouped = df['data1'].groupby([df['key1']])
    print(grouped.mean())
    # 多个分组键
    means = df['data1'].groupby([df['key1'], df['key2']]).mean()
    print('mean1:')
    print(means)
    print(means.unstack())
#     分组键为数组
    states = np.array(['Ohio', 'Ohio', 'California', 'California', 'Ohio'])
    years = np.array([2000, 2001, 2002, 2003, 2004])
    means = df['data1'].groupby([states, years]).mean()
    print('means2:')
    print(means)
    print(means.unstack())
#    列名作为
    means = df.groupby(['key1','key2']).mean()
    print(means)
    print(means.unstack())
    print(df.groupby(['key1', 'key2']).size())
#
#   对分组进行迭代
    print('对分组进行迭代')
    for name, group in df.groupby('key1'):
        print('-----------')
        print(name)
        print('---')
        print(group)
        print('-----------')

    for (key1,key2), group in df.groupby(['key1', 'key2']):
        print('-----------')
        print(key1, key2)
        print('---')
        print(group)
        print('-----------')
#     自作成字典
    pieces = dict(list(df.groupby('key1')))
    print(pieces['b'])

#     根据列进行分组
    print('根据列进行分组')
    print(df.dtypes)
    print(type(df.dtypes))
    grouped = df.groupby(df.dtypes, axis=1)
    print(dict(list(grouped)))

    print(df.groupby('key1')['data1'])
    print(df['data1'].groupby(df['key1']))
    print(df.groupby('key1')['data2'])
    print(df['data2'].groupby(df['key1']))
    print(df.groupby('key1')['data2'].mean())
    print(df['data2'].groupby(df['key1']).mean())


# 通过字典或series进行分组
def dict_series_groupby():
    df = DataFrame(np.random.randn(5,5),
                   columns={'a', 'b', 'c', 'd', 'e'},
                   index={'jow', 'steve', 'wes', 'jim', 'travis'})
    df.ix[2:3, ['b', 'c']] = np.nan
    print(df)
    mapping = {'a': 'red', 'b': 'red', 'c': 'blue', 'd': 'blue', 'e': 'red', 'f': 'orange'}
    by_columns = df.groupby(mapping, axis=1)
    print(by_columns.sum())

    map_series = Series(mapping)
    print(map_series)
    by_columns = df.groupby(map_series, axis=1).count()
    print(by_columns)


def fun_groupby():
    df = DataFrame(np.random.randn(5, 5),
                   columns={'a', 'b', 'c', 'd', 'e'},
                   index={'jow', 'steve', 'wes', 'jim', 'travis'})

    df.ix[2:3, ['b', 'c']] = np.nan
    print(df)
    print(df.groupby(len).sum()) # 根据index长度进行分类

#     同数组，列表，字典，series进行混用
    key_list = ['one', 'one', 'one', 'two', 'two']
    print(df.groupby([len, key_list]).min())


# 根据索引级别分组
def index_level_group():
    columns = pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'],
                                         [1, 3, 4, 1, 3]], names=['city', 'tenor'])
    df = DataFrame(np.random.randn(4, 5), columns=columns)
    print(df)
    print(df.groupby(level='city', axis=1).count())


# 分位数
def quantile():
    df = DataFrame({'key1': ['a', 'a', 'b', 'b', 'a'],
                    'key2': ['one', 'two', 'three', 'four', 'five'],
                    'data1': np.random.randn(5),
                    'data2': np.random.randn(5)})
    grouped = df.groupby('key1')
    print(grouped['data1'].quantile(0.9))
#     使用自定义函数
    print(grouped.agg(peak_to_peak))


def peak_to_peak(arr):
    return arr.max() - arr.min()


if __name__ == '__main__':
    # groupby1()
    # dict_series_groupby()
    # fun_groupby()
    # index_level_group()
    quantile()
