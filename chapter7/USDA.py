import json
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from utils.logger.logger import logger

logger.info('load data')
db = json.load(open('data/foods-2011-10-03.json'))
print(len(db))
print(db[0].keys())

print(db[0]['nutrients'][0])
nutrients = DataFrame(db[0]['nutrients'])
print(nutrients[:7])
info_keys = ['id', 'description', 'manufacturer', 'group']
info = DataFrame(db, columns=info_keys)
print(info)
print(info[:5])
print(info.describe())
print(info.info())

print(pd.value_counts(info.group)[:10])
logger.info('整合 nutrients')
nutrients =[]
for item in db:
    fnuts = DataFrame(item['nutrients'])
    fnuts['id'] = item['id']
    nutrients.append(fnuts)
nutrients = pd.concat(nutrients, ignore_index=True)
print(nutrients.info())
logger.info('去重')
print(nutrients.duplicated().sum())
nutrients = nutrients.drop_duplicates()
print(nutrients.info())
logger.info('列重命名')
col_mapping = {'description':'food',
               'group': 'fgroup'}
info = info.rename(columns=col_mapping, copy= False)
print(info.info())
col_mapping = {'description': 'nutrient',
               'group': 'ngroup'}
nutrients = nutrients.rename(columns = col_mapping, copy = False)
print(nutrients.info())

logger.info('info 和 nutrients进行合并')
ndata = pd.merge(nutrients,info,on ='id', how='outer')
print(ndata.info())
print(ndata.ix[3000])

result = ndata.groupby(['nutrient','fgroup'])['value'].quantile(0.5)
result['Zinc, Zn'].plot(kind='bar')
by_nutrient = ndata.groupby(['nutrient','ngroup'])
get_maixmum = lambda x: x.xs(x.value.idxmax())
get_min = lambda x: x.xs(x.value.idxmin())
max_foods = by_nutrient.apply(get_maixmum)[['value','food']]
max_foods.food = max_foods.food.str[:50]
print(max_foods)
print(max_foods.ix[10]['food'])