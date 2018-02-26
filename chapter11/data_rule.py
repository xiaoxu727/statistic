import random; random.seed(0)
import string
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from datetime import time
# import pandas.io.data as web
from statsmodels.formula.api import ols
from pandas_datareader import data as web

def data():
    raw_prices = pd.read_csv('data/stock_px.csv', parse_dates=True, index_col=0)
    print(raw_prices[:2])
    prices = raw_prices[['AAPL', 'JNJ', 'SPX', 'XOM']]
    # prices.index = raw_prices.index
    print(prices[:2])
    raw_volume = pd.read_csv('data/volume.csv', parse_dates=True, index_col=0)
    volume = raw_volume[['AAPL', 'JNJ', 'SPX', 'XOM']]
    # volume.index = raw_volume.index
    return prices, volume


def proc():
    prices, volume = data()
    print(prices * volume)

    print(volume.sum())
    vwap = (prices * volume).sum()/volume.sum()
    print(vwap)

def reindex_resample():
    ts1 = Series(np.random.randn(3), index=pd.date_range('2016-6-13', periods=3, freq='W-WED'))
    print('ts1')
    print(ts1)
    print('-------------------------')
    print(ts1.resample('B').mean())
    print('-------------------------')
    print(ts1.resample('B').ffill())

    dates = pd.DatetimeIndex(['2016-6-12', '2016-6-17', '2016-6-18', '2016-6-22', '2016-6-27', '2016-6-29'])
    ts2 = Series(np.random.randn(6), index= dates)
    print('---------ts2----------------')
    print(ts2)
    print(ts1.reindex(ts2.index))
    print(ts1.reindex(ts2.index).ffill())

    gdp = Series([1.78, 1.34, 1.323, 4.43, 6.3, 2.33],
                 index=pd.period_range('2000Q1', periods=6, freq='Q-SEP'))
    print(gdp)

    infl = Series([0.20, 0.11, 0.24, 0.53],
                  index=pd.period_range('2000', periods=4, freq='A-DEC'))
    print(infl)

    infl_q = infl.asfreq('Q-SEP', how='end')
    print(infl_q)
    print(infl_q.reindex(gdp.index))
    print(infl_q.reindex(gdp.index).ffill())


def time_selection():
    rng = pd.date_range('2012-06-01 09:30', '2012-06-01 16:59', freq='T')
    rng = rng.append([rng+pd.offsets.BDay(i) for i in range(1, 4)])
    ts = Series(np.arange(len(rng), dtype=float), index=rng)
    print(ts)

    print(ts[time(10, 0)])
    print(ts.at_time(time(10, 0)))

    print(ts.between_time(time(10, 0), time(10, 1)))

    indexer = np.sort(np.random.permutation(len(ts))[700:])

    irr_ts = ts.copy()
    irr_ts[indexer] = np.nan
    print(irr_ts['2012-06-01 09:30':'2012-06-01 10:00'])

    selection = pd.date_range('2012-06-01 10:00', periods=4, freq='B')
    print(irr_ts.asof(selection))


def multi_data():
    data1 = DataFrame(np.ones((6, 3), dtype=float),
                      columns=['a', 'b', 'c'],
                      index=pd.date_range('6/12/2012', periods=6))
    data2 = DataFrame(np.ones((6, 4),dtype=float)*2,
                      columns=['a', 'b', 'c','d'],
                      index=pd.date_range('6/13/2012', periods=6))

    spliced = pd.concat([data1.ix[:'2012-06-14'], data2.ix['2012-06-15':]])
    print(spliced)

    spliced_filled = spliced.combine_first(data2)
    print(spliced_filled)

    cp_spliced = spliced.copy()
    spliced.update(data2, overwrite=False)
    print(spliced)

    cp_spliced.update(data2, overwrite=True)
    print(cp_spliced)

    cp_spliced[['a', 'c']] = data1[['a', 'c']]
    print(cp_spliced)
    # print(spliced.update(data2, overwrite=True))

def returns():
    price = web.get_data_yahoo_actions('AAPL', '2011-01-01','2012-07-27')['Adj Close']
    print(price[-5:])
    returns = price.pct_change()
    ret_index = (1+returns).cumprod()
    print(ret_index)

N = 1000
def rands(n):
    choices = string.ascii_uppercase
    return ''.join([random.choice(choices) for _ in range(n)])
def zscore(group):
    return (group -group.mean()) / group.std()

M = 500
def get_tickers():
    tickers = np.array([rands(5) for _ in range(N)])
    # print(tickers)
    # print(tickers[:M])
    df = DataFrame({'Momentum': np.random.randn(M) / 200 + 0.03,
                   'Value': np.random.randn(M) / 200 + 0.08,
                   'ShortInterest': np.random.randn(M) / 200 - 0.02},
                   index=tickers[:M])
    # print(tickers)
    ind_names = np.array(['FINANCIAL', 'TEC'])
    sampler = np.random.randint(0, len(ind_names), N)
    industries = Series(ind_names[sampler], index=tickers, name='industry')
    print(industries)
    by_industry = df.groupby(industries)
    print(by_industry.mean())
    print(by_industry.describe())
    df_stand = by_industry.apply(zscore)
    print(df_stand)
    # print(df_stand.describe())
    print(df_stand.groupby(industries).agg(['mean', 'std']))

    ind_rank = by_industry.rank(ascending=False)
    print(ind_rank.groupby(industries).agg(['min', 'max']))
    print(ind_rank.apply(lambda x: zscore(x.rank())))

    fac1, fac2, fac3 = np.random.randn(3, 1000)
    ticker_subset = tickers.take(np.random.permutation(N)[:1000])
    # print(ticker_subset)
    port = Series(0.7*fac1 - 1.2 * fac2 + 0.3 * fac3 + np.random.rand(1000), index=ticker_subset)
    factors = DataFrame({'f1': fac1, 'f2': fac2, 'f3': fac3}, index=ticker_subset)
    print(factors.corrwith(port))
    print(pd.stats.ols(y=port, x=factors).beta)

def group_analysis():
    return



if __name__ == '__main__':
    # prices, volume = data()
    # print(prices[:3])
    # print(volume[:3])
    # proc()
    # reindex_resample()
    # print(time(10, 0))
    # time_selection()
    # multi_data()
    # returns()
    # print(np.arange(18).reshape((6, 3)))
    # data1 = DataFrame(np.arange(3 * 6).reshape((6, 3)),
    #                   columns=['a', 'b', 'c'],
    #                   index=pd.date_range('6/12/2012', periods=6))
    #
    # print(data1)
    # print(data1.pct_change())

    # print(rands(3))
    get_tickers()
    # test = np.array([1, 2, 3])
    # print(test.take([0]))
    # print(np.random.permutation(10))