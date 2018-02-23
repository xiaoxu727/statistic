import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import chapter9.tips as tp

def piovt():
    tips = tp.load_data()
    print(tips.info())
    print(tips[-4:])
    print(tips.pivot_table(index =['sex','smoker']))

    print(tips.pivot_table(['tip_pct', 'size'], index=['sex','day'], columns='smoker'))
    print(tips.pivot_table(['tip_pct', 'size'], index=['sex','day'], columns='smoker', margins=True))
    print(tips.pivot_table(['tip_pct', 'size'], index=['sex','day'], columns='smoker', margins=True, aggfunc=len))
    print(tips.pivot_table(['tip_pct', 'size'], index=['sex','day'], columns='smoker', margins=True, aggfunc='sum', fill_value=0))


def crosstab():
    tips = tp.load_data()
    print(pd.crosstab([tips.time, tips.day], tips.smoker, margins=True))



if __name__ == '__main__':
    # piovt()
    crosstab()