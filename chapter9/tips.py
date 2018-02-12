import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import chapter9.groupby as gp



def load_data():
    tips = pd.read_csv('../chapter8/data/tips.csv')
    tips['tip_pct'] = tips['tip'] / tips['total_bill']
    return tips
    # print(tips)


#多个函数调用
def funs_call():
    tips = load_data()
    grouped = tips.groupby(['sex', 'smoker'])
    grouped_pct = grouped['tip_pct']
    print(grouped_pct.aggregate('mean'))

    print(grouped_pct.aggregate(['mean', 'std', gp.peak_to_peak]))

    print(grouped_pct.aggregate([('m1','mean'),('m2', 'std'), ('m3', gp.peak_to_peak)]))

    functions = ['count', 'mean', 'max']
    result = grouped['tip_pct', 'total_bill'].aggregate(functions)
    print(result)
    print(result['tip_pct'])

    ftuples = [('m1', 'mean'), ('m2', np.var)]
    print(grouped['tip_pct', 'total_bill'].agg(ftuples))
#     不同的列用不同函数，传入字典
    print(grouped.agg({'tip': np.max, 'size': 'sum'}))
    print(grouped.agg({'tip': [np.max, 'min'], 'size': 'sum'}))
#     以无索引的形式返回聚合函数
    print(tips.groupby(['sex', 'smoker']).mean())
    print(tips.groupby(['sex', 'smoker'], as_index=False).mean())


if __name__ == '__main__':
    # load_data()
    funs_call()