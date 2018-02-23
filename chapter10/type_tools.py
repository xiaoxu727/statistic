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
    period()