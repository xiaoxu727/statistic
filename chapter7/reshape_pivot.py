# 重塑和轴向旋转
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from utils.logger.logger import logger

data = DataFrame(np.arange(6).reshape(2,3),index=pd.Index(['Ohio','Colorado'],
                 name='state'),columns=pd.Index(['one','two','three'],name='number'))
print(data)
logger.info("columns to rows by stack(),get series")
result = data.stack()
print(result)

logger.info("rows to columns by unstack() get dataframe")

print(result.unstack())

logger.info("旋转根据分层级别的名称或编号")

print(result.unstack(0))
print(result.unstack('state'))

s1 = Series([1,2,3,4],index=['a','b','c','d'])
s2 = Series([4,5,6],index=['c','d','e'])
data2 = pd.concat([s1,s2], keys=['one','two'])
print(data2)
print(data2.unstack())
print(data2.unstack().stack(dropna=False))
logger.info("作为旋转轴的级别最低")
df = DataFrame({'left': result, 'right': result+5},
               columns=pd.Index(['left','right'],name='side'))
print(df)
print(df.unstack('state'))
print(df.unstack('state').stack('side'))

# 将长格式旋转为宽格式



