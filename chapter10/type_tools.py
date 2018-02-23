from datetime import datetime
from datetime import timedelta
from dateutil.parser import parse
import pandas as pd
from pandas import Series
import numpy as np
from pandas import  DataFrame
from pandas.tseries.offsets import Hour, Minute
from pandas.tseries.offsets import Day, MonthEnd
import pytz
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore


def basic():
    now = datetime.now()
    print(now)
    print(now.year)
    print(now.month)
    print(now.day)
    # 时间差
    delta = datetime(2018, 2, 22) - datetime(2016, 2, 11,  8, 15)
    print(delta)
    print(delta.days)
    print(delta.seconds)

    start = datetime(2018, 2, 22)
    end = start + timedelta(12)
    print(end)
    end = start - 2 * timedelta(12)
    print(end)


def str_to_datetime():
    stamp = datetime(2011, 1, 3)
    print(stamp)
    print(str(stamp))
    print(stamp.strftime('%Y-%m-%d'))
    print(stamp.strftime('%y-%m-%d'))
    value = '2011-01-03'
    print(datetime.strptime(value, '%Y-%m-%d'))
    strs = ['2/22/2018', '2/1/2018']
    [print(datetime.strptime(str, '%m/%d/%Y')) for str in strs]

#     parse
    print(parse('2018-2-23'))
    print(parse('Jan 31, 2018 10:45 PM'))
    print(parse('22/2/2018', dayfirst=True))

#  pd.to_datetime
    print(pd.to_datetime(strs))
    print(pd.to_datetime(strs+[None]))
    print(pd.isnull(pd.to_datetime(strs+[None])))


def timeseries_basic():
    dates = [datetime(2018, 1, 1), datetime(2018, 1, 2),datetime(2018, 1, 3), datetime(2018, 1, 4),
             datetime(2018, 1, 5), datetime(2018, 1, 6), datetime(2018, 1, 7)]
    ts = Series(np.random.randn(7), index=dates)
    print(ts)
    print(type(ts))
    print(ts.index)
    print(ts + ts[::2])
    print(ts.index[0])


def index():
    dates = [datetime(2018, 1, 1), datetime(2018, 1, 2), datetime(2018, 1, 3), datetime(2018, 1, 4),
             datetime(2018, 1, 5), datetime(2018, 1, 6), datetime(2018, 1, 7)]
    ts = Series(np.random.randn(7), index=dates)
    stamp = ts.index[0]
    print(ts[stamp])
    print(ts['2018/1/1'])
    long_ts = Series(np.random.randn(10000), index=pd.date_range('1/1/2018',periods=10000))
    # print(long_ts)
    print(long_ts['2018'])
    print(long_ts[datetime(2045, 1, 1):])
    print(long_ts['20450101':'20450201'])
    print(long_ts.truncate(after='20450301'))
    dates = pd.date_range('1/1/2000', periods=100, freq='W-WED')
    long_df = DataFrame(np.random.randn(100 ,4), index=dates, columns=['Colorado', 'Texas', 'New York', 'Ohio'])
    print(long_df.ix['2001-5'])


def repeat():
    dates = pd.DatetimeIndex(['1/2/2001', '1/2/2001', '1/2/2001', '1/2/2001', '1/2/2001', '1/3/2001'])
    dup_ts = pd.Series(np.arange(6), index=dates)
    print(dup_ts)
    print(dup_ts.is_unique)
    print(dup_ts['1/2/2001'])
    grouped = dup_ts.groupby(level=0)
    print(grouped.mean())
    print(grouped.count())

def reshample():
    dates = [datetime(2018, 1, 1), datetime(2018, 1, 2), datetime(2018, 1, 3), datetime(2018, 1, 4),
             datetime(2018, 1, 10), datetime(2018, 1, 14), datetime(2018, 1, 17)]
    ts = Series(np.random.randn(7), index=dates)
    print(ts)
    resample = ts.resample('D').asfreq()
    # print(ts.resample('D'))
    print(resample)
    # print(resample.sum())


def date_range():
    index = pd.date_range('1/1/2018', '2/22/2018')
    print(index)
    print(pd.date_range(start='1/2/2018', periods=10))
    print(pd.date_range(end='1/2/2018', periods=10))
    print(pd.date_range('2018/1/1', '2018/3/22', freq='BM'))
    print(pd.date_range('2018/1/1 12:21:22', periods=5))
    print(pd.date_range('2018/1/1 12:21:22', periods=5, normalize=True))


def frequency():
    hour = Hour(1)
    four_hours = Hour(4)
    print(hour)
    print(four_hours)
    one_half_hour = Hour() + Minute(30)
    print(one_half_hour)
    print(pd.date_range('2018-2-1', '2018-2-2', freq='2h'))

    print(pd.date_range('1/1/2018', '2/22/2018', freq='WOM-3MON')) #每月第三个星期一


def shift():
    ts = Series(np.random.randn(4), index=pd.date_range('2018-1-1', periods=4, freq='M'))
    print(ts)
    print(ts.shift(2))
    print(ts.shift(-2))
    print(ts/ts.shift(1)-1) #百分比变化
    print(ts.shift(2, freq='M'))
    print(ts.shift(2, freq='D'))
    print(ts.shift(2, freq='90min'))

    now = datetime.now()
    print(now + MonthEnd())

    offset = MonthEnd()
    print(offset.rollback(now))
    print(offset.rollforward(now))


def timezone():
    print(pytz.common_timezones[:])
    print(pytz.timezone('Asia/Shanghai'))
    rng = pd.date_range('2/22/2018 16:30', periods=6, freq='D')
    ts = Series(np.random.randn(len(rng)), index=rng)
    print(ts)
    print(ts.index.tz)
    ts_utc = ts.tz_localize('UTC')
    print(ts_utc.index.tz)
    print(ts_utc.tz_convert('Asia/Shanghai'))
#     不同时区之间运算
    rng = pd.date_range('2/22/2018 16:30', periods=10, freq='B')
    ts = Series(np.random.randn(len(rng)), index=rng)
    print(ts)
    ts1 = ts.tz_localize('Asia/Shanghai')
    ts2 = ts.tz_localize('Europe/London')
    print(ts1)
    print(ts2)
    print((ts1+ts2).index)


def period():
    p = pd.Period(2018, freq='A-DEC')
    print(p)
    print(p + 5)
    print(pd.Period(2014, freq='A-DEC') - p)
    rng = pd.period_range('1/1/2001', '22/12/2001', freq='M')
    print(rng)
    print(Series(np.random.randn(len(rng)), index=rng))

    values = ['2001Q3', '2001Q2', '2001Q1']
    index = pd.PeriodIndex(values, freq='Q-DEC')
    print(index)

    p = pd.Period('2007', freq='A-DEC')
    print(p.asfreq('M', how='start'))
    print(p.asfreq('M', how='end'))

    p = pd.Period('2007', freq='A-JUN')
    print(p.asfreq('M', how='start'))
    print(p.asfreq('M', how='end'))

    p = pd.Period('2007-08', 'M')
    print(p.asfreq('A-JUN'))
    print(p)

    rng = pd.period_range('2006', '2009', freq='A-DEC')
    ts = Series(np.random.randn(len(rng)), index=rng)
    print(ts)

    print(ts.asfreq('M', how='start'))
    print(ts.asfreq('B', how='end'))

    p = pd.Period('2012Q4', freq='Q-JAN')
    print(p)
    print(p.asfreq('D', 'start'))
    print(p.asfreq('D', 'end'))

    p4pm = (p.asfreq('B', 'end')-1).asfreq('T', 's') + 16 * 60
    print(p4pm)

    rng = pd.period_range('2011Q3', '2012Q4', freq='Q-JAN')
    ts = Series(np.arange(len(rng)), index=rng)


# error!!
def timestamp_2_period():
    rng = pd.period_range('2001/1/1', periods=3, freq='M')
    ts = Series(np.random.randn(3), index=rng)
    print(ts)
    print(type(ts))
    pts = Series.to_period(ts)
    print(pts)


def periodIndex():
    data = pd.read_csv('../chapter8/data/macrodata.csv')
    print(data.year)
    print(data.quarter)

    index = pd.PeriodIndex(year=data.year, quarter=data.quarter, freq='Q-DEC')

    print(index)
    data.index = index
    print(data.infl)


def resample_frequency():
    rng = pd.date_range('1/1/2000', periods=100, freq='D')
    ts = Series(np.arange(100), index=rng)

    print(ts)
    print(ts.resample('M').mean())
    print(ts.resample('M', kind='period').mean())

#     降采样
    rng = pd.date_range('1/1/2000', periods=12, freq='T')
    ts = Series(np.arange(12), index=rng)
    print(ts)
    print(ts.resample('5min').sum())
    print(ts.resample('5min', closed='left').sum())
    print(ts.resample('5min', closed='right').sum())
    print(ts.resample('5min', closed='right', label='left').sum())
    print(ts.resample('5min', closed='right', label='right').sum())
    print(ts.resample('5min', closed='right', label='right', loffset='-1s').sum())

# OHLC重新采样
    print(ts.resample('5min').ohlc())
    # groupby 重采样
    print(ts.groupby(lambda x: x.month).mean())
    print(ts.groupby(lambda x: x.weekday).mean())

# 升采样和插值
    frame = DataFrame(np.random.randn(2, 4),
                      index=pd.date_range('1/1/2000', periods=2, freq='W-WED'),
                      columns=['Colorado', 'Texas', 'New York', 'Ohio'])
    print(frame)
    df_daily = frame.resample('D')
    print(df_daily.mean())
    print(frame.resample('D').ffill(limit=2))

# 通过时期进行重新采样
    frame = DataFrame(np.random.randn(23, 4),
                      index=pd.date_range('1-2000', '12-2001', freq='M'),
                      columns=['Colorado', 'Texas', 'New York', 'Ohio'])
    print(frame)
    annual_frame = frame.resample('A-DEC').mean()
    print(annual_frame)

    print(annual_frame.resample('Q-DEC').ffill())
    print(annual_frame.resample('Q-DEC', convention='start').ffill())


def time_draw():
    close_px_all = pd.read_csv('../chapter9/data/stock_px.csv', parse_dates=True, index_col=0)
    close_px = close_px_all[['AAPL', 'MSFT', 'XOM']]
    close_px = close_px.resample('B').ffill()
    print(close_px)
    # close_px['AAPL'].plot()
    # close_px.ix['2009'].plot()
    # close_px['AAPL'].ix['01-2011':'03-2011'].plot()
    # appl_q = close_px['AAPL'].resample('Q-DEC').ffill()
    # appl_q.ix['2009':].plot()
    # 移动窗口函数
    # close_px.AAPL.plot()
    # pd.rolling_mean(close_px.AAPL, 250).plot()
    #
    # appl_std250 = pd.rolling_std(close_px.AAPL, 250, min_periods=50)
    # print(appl_std250[5:12])
    # appl_std250.plot()

    # pd.rolling_mean(close_px.AAPL, 10).plot(logy=True)
    # pd.rolling_mean(close_px.AAPL, 30).plot(logy=True)
    # pd.rolling_mean(close_px.AAPL, 60).plot(logy=True)
    # pd.rolling_mean(close_px.AAPL, 100).plot(logy=True)
    # close_px.AAPL.plot()
    # # 指数加权函数
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=(12,7))
    appl_px = close_px.AAPL['2005': '2009']
    # ma60 = pd.rolling_mean(appl_px, 60, min_periods=50)
    # ma60 = Series.rolling(appl_px, 60, min_periods=50).mean()
    # # ewa60 = pd.ewma(appl_px, span=60)
    # ewa60 = Series.ewm(appl_px, span=60).mean()
    #
    # appl_px.plot(style='k-', ax=axes[0])
    # ma60.plot(style='k--', ax=axes[0])
    # appl_px.plot(style='k-', ax=axes[1])
    # ewa60.plot(style='k--', ax=axes[1])
    # axes[0].set_title('Simple MA')
    # axes[1].set_title('Expontially-weighted MA')

    spx_px = close_px_all['SPX']
    spx_px_sets = spx_px/spx_px.shift(1)-1

    returns = close_px.pct_change()
    # corr = pd.rolling_corr(returns.AAPL, spx_px_sets, 125, min_periods=100)
    # corr = Series.rolling(returns.AAPL,window=125, min_periods=100).corr(spx_px_sets)
    # corr.plot(ax=axes[0])
    # # corr = pd.rolling_corr(returns, spx_px_sets, 125, min_periods=100)
    # corr = DataFrame.rolling(returns,window=125, min_periods=100).corr(spx_px_sets)
    # corr.plot(ax=axes[1])
    score_at_2percent = lambda x: percentileofscore(x, 0.02)
    # result = pd.rolling_apply(returns.AAPL, 250, score_at_2percent)
    result = Series.rolling(returns.AAPL, 250).apply(score_at_2percent)
    result.plot()

    plt.show()


if __name__ == '__main__':
    # str_to_datetime()
    # timeseries_basic()
    # index()
    # repeat()
    # reshample()
    # date_range()
    # frequency()
    # shift()
    # timezone()
    # period()
    # timestamp_2_period()
    # periodIndex()
    # resample_frequency()
    time_draw()