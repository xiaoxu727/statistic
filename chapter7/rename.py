# 重命名索引
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from utils.logger.logger import  logger
logger.info('重名名')
data = DataFrame(np.arange(12).reshape(3,4),
                 index=['Ohio','Colorado','New York'],
                 columns=['one','two','three','four'])
print(data.index.map(str.upper))
data.index = data.index.map(str.upper)
print(data)

print(data.rename(columns= str.upper))
print(data.rename(index = str.title,columns= str.upper))
print(data)
print(data.rename(index={'OHIO':'INDIANA'},columns = {'three':'peekboo'}))
logger.info('inplace=True进行标签赋值')
print(data)
_ = data.rename(index={'OHIO':'INDIANA'},inplace=True)
print(data)

logger.info('离散化和面元划分')

ages = [12,13,34,54,6,26,89,35,54]
bins =[1,18,30,40,50,80,90]
cats = pd.cut(ages,bins)
print(cats)
print(cats.codes)
print(cats.categories)
print(pd.value_counts(cats))
logger.info('right=False设置右面为开端')
print(pd.cut(ages,bins,right=False))
logger.info('传入不确切的边界，传入分类的数量，根据数据最大值和最小值进行划分')
print(pd.cut(ages,4,precision = 1))

logger.info('qcut,获得等数据量的数据')
data = np.random.randn(1000)
cats = pd.qcut(data,4)
print(cats)
print(pd.value_counts(cats))
logger.info('检测和过滤异常值')
print(np.random.seed(12345))
data = DataFrame(np.random.randn(1000,4))
print(data.describe())
col = data[3]
print(col[np.abs(col)>3])
print(data[(np.abs(data)>3).any(1)])
data[np.abs(data)>3]=np.sign(data)*3
print(data.describe())

logger.info('排列和随机采样')

df = DataFrame(np.arange(5*4).reshape(5,4))
sampler = np.random.permutation(5)
print(df)
print(sampler)
print(df.take(sampler))
print(df.take(np.random.permutation(len(df))[:3]))
logger.info('np.random.randint()')
bag = np.arange(10)
sampler = np.random.randint(0,len(bag),size=10)
print(bag)
print(sampler)
print(bag.take(sampler))

logger.info('计算指标/哑变量')


df = DataFrame({'key':['b','b','a','c','b'],
                'data1':range(5)})
print(df)
print(pd.get_dummies(df['key']))
dummies = pd.get_dummies(df['key'],prefix='key')
df_with_dummy = df[['data1']].join(dummies)
print(df_with_dummy)
logger.info('一行属于多类')
mnames = ['movie_id','title','genres']
movies = pd.read_table('../chapter2/data/movielens/movies.dat', sep='::', header=None, names=mnames, engine='python')
print(movies[:10])

genre_iter = (set(x.split('|')) for x in movies['genres'])
# for x in genre_iter:
#     print(x)
# print(genre_iter)
genres = set()
for x in genre_iter:
    genres = genres.union(x)
genres = sorted(genres)
print(genres)
dummies = DataFrame(np.zeros((len(movies),len(genres))),columns=genres)
print(dummies)
for i, gen in enumerate(movies['genres']):
    # print(i)
    # print(gen)
    dummies.ix[i, gen.split('|')] = 1
print(dummies)
movies_windic = movies.join(dummies.add_prefix('Genre_'))
logger.info('与movies联合起来')
print(movies_windic)
print(movies_windic.ix[0])
