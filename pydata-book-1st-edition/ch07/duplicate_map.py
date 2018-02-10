# 数据转换
import numpy as np
import pandas as pd
from pandas import DataFrame
from utils.logger.logger import logger
logger.info('移除重复数据')
df = DataFrame({'k1':['one']*3 + ['two']*4,
               'k2':[1,1,2,2,3,4,4]})
logger.info('source')
print(df)
logger.info('duplicate')
print(df.duplicated())
logger.info('drop_duplicates')
print(df.drop_duplicates())

df['v1']=range(7)
print(df.drop_duplicates(['k1']))

logger.info('取最后一个')
print(df.drop_duplicates(['k1', 'k2'],keep='last'))

logger.info('利用函数或映射进行数据转换')
data = DataFrame({'food':['bacon','pulled pork','bacon','Pastrami',
                          'corned beef','Bacon','pastrami','honey ham','nova lox'],
                  'ounces':[4,3,12,6,7.5,8,3,5,6]})
meat_to_animal={
    'bacon':'pig',
    'pulled pork':'pig',
    'pastrami':'cow',
    'corned beef':'cow',
    'honey ham':'pig',
    'nova lox':'salmon'
}
data['animal'] = data['food'].map(str.lower).map(meat_to_animal)
print(data)
print(data['food'].map(lambda x:meat_to_animal[x.lower()]))

logger.info('转换值')
