# 分组加权平均数和相关系数
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import statsmodels.api as sm


def test1():
    df = DataFrame({'category': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
                    'data': np.random.randn(8),
                    'weights': np.random.randn(8)})

    print(df)
    grouped = df.groupby('category')
    get_wavg = lambda x: np.average(x['data'], weights=x['weights'])
    print(grouped.apply(get_wavg))

def yahoo_finance():
    close_px = pd.read_csv('data/stock_px.csv', parse_dates=True, index_col=0)
    print(close_px[:4])
    print(close_px.info())
    rets = close_px.pct_change().dropna()
    print('pct_change')
    print(rets[-4:])
    spx_corr = lambda x: x.corrwith(x['SPX'])
    by_year = rets.groupby(lambda x: x.year)
    print(by_year)
    res = by_year.apply(spx_corr)
    print(res[-4:])
    print(by_year.apply(lambda g: g['AAPL'].corr(g['MSFT'])))

    print(by_year.apply(regress, 'AAPL', ['SPX']))


def regress(data, yvar, xvars):
    Y = data[yvar]
    X = data[xvars]
    X['intercept']=1
    result = sm.OLS(Y, X).fit()
    return result.params

if __name__ == '__main__':
    yahoo_finance()
