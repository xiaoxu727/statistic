import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import chapter9.tips as tp


def group_merge():
    df = DataFrame({'key1': ['a', 'a', 'b', 'b', 'a'],
                    'key2': ['one', 'two', 'three', 'four', 'five'],
                    'data1': np.random.randn(5),
                    'data2': np.random.randn(5)})
    means = df.groupby('key1').mean()
    print(means)
    print(pd.merge(df, means, left_on='key1', right_index=True))

def transform():
    people = DataFrame(np.random.randn(5, 5),
                   columns={'a', 'b', 'c', 'd', 'e'},
                   index={'jow', 'steve', 'wes', 'jim', 'travis'})
    key = np.array(['one', 'two', 'one', 'two', 'one'])
    mean = people.groupby(key).mean()
    print(mean)
    print(people.groupby(key).transform(np.mean))

    demeaned = people.groupby(key).transform(demean)
    print(demeaned)
    print(demeaned.groupby(key).mean())


def demean(arr):
    return arr - arr.mean()


def apply():
    tips = tp.load_data()
    print(tips)
    print(tips.groupby('smoker').apply(top))
    print(tips.groupby('smoker', group_keys=False).apply(top)) #禁止分组键
    # print(tips.apply(top)) #禁止分组键
    print(tips.sort_values(by='tip_pct')[-5:])
    print('---------clone--------')
    print(tips.apply(clone))
    print(tips.groupby(['smoker', 'day']).apply(top, n=1, columns='total_bill'))
    print(tips.groupby('smoker')['tip_pct'].describe())
    print(tips.groupby('smoker')['tip_pct'].describe().unstack())


# 分位数和桶分析
def bucket_quantile():
    frame = DataFrame({'data1': np.random.randn(1000),
                       'data2': np.random.randn(1000)})
    factor = pd.cut(frame.data1, 4)
    print(factor[:10])
    grouped = frame.data2.groupby(factor).apply(get_stats).unstack()
    print(grouped)

    grouping = pd.qcut(frame.data1, 10, labels=False)
    print(frame.data2.groupby(grouping).apply(get_stats).unstack())


def get_stats(group):
    return {'min': group.min(), 'max:': group.max(),
            'count': group.count(), 'mean': group.mean()}



def clone(df, columns=1):
    return df.iloc[[1]]


def top(df, n=5, columns='tip_pct'):
    return df.sort_values(by=columns)[-n:]


if __name__ == '__main__':
    # group_merge()
    # transform()
    # apply()
    bucket_quantile()