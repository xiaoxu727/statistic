import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from datetime import time
# import pandas.io.data as web
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
    print(np.arange(18).reshape(6, 3))
    data1 = DataFrame(np.arange(3 * 6).reshape(3, 6),
                      columns=['a', 'b', 'c'],
                      index=pd.date_range('6/12/2012', periods=6))
    # print(data1.pct_change())