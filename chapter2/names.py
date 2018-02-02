import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt

def load_data():
    # names1880 = pd.read_csv('data/names/yob1880.txt',names =['name','sex','births'])
    # print names1880.groupby('sex').births.sum()
    # print names1880.info()
    # print names1880[:10]
    years = range(1880,2011)
    pices = []
    columns = ['name','sex','births']
    for year in years:
        path = 'data/names/yob%d.txt'%year
        frame = pd.read_csv(path,names = columns)
        frame['year'] = year
        pices.append(frame)
    names = pd.concat(pices,ignore_index= True)
    print names[:10]
    total_birts = names.pivot_table('births',index = 'sex',columns = 'year',aggfunc=sum)
    total_birts.plot(title='total births by sex and year')
    print total_birts.tail()
    names = names.groupby(['year','sex']).apply(add_prop)
    grouped = names.groupby(['year','sex'])
    top1000= grouped.apply(get_top1000)
    boys = top1000[top1000.sex=='M']
    girls = top1000[top1000.sex=='F']
    # names
    total_birts = top1000.pivot_table('births',index='year',columns='name',aggfunc=sum)
    subset = total_birts[['John','Harry','Mary','Marilyn']]
    subset.plot(subplots = True, figsize=(12,10),grid=False,title='Number birth per year')
    #
    print top1000.info()
    table = top1000.pivot_table('prop',index='year',columns='sex',aggfunc=sum)
    table.plot(title='sum of table1000.prop by year and sex',yticks=np.linspace(0,1.2,13),xticks=range(1880,2020,10))

    df= boys[boys.year==2010]
    prop_cumsum= df.sort_values(by='prop',ascending = False).prop.cumsum()
    # print prop_cumsum[:10]
    print prop_cumsum.searchsorted(0.5)


    df = boys[boys.year == 1900]
    prop_cumsum = df.sort_values(by='prop', ascending=False).prop.cumsum()
    # print prop_cumsum[:10]
    print prop_cumsum.searchsorted(0.5)
    diversity = top1000.groupby(['year','sex']).apply(get_quantile_count)
    diversity= diversity.unstack('sex')
    print diversity[:10]
    get_last_letter = lambda x:x[-1]
    last_letters = names.name.map(get_last_letter)
    last_letters.name = 'last_letter'
    table = names.pivot_table('births',index = last_letters,columns=['sex','year'],aggfunc=sum)
    subtable = table.reindex(columns=[1910,1960,2010],level  ='year')
    print subtable.head()
    print subtable.sum()
    letter_prop = subtable/subtable.sum().astype(float)
    fig,axes = plt.subplots(2,1,figsize=(10,8))
    letter_prop['M'].plot(kind='bar',rot=0,ax = axes[0],title = 'Male')
    letter_prop['F'].plot(kind='bar',rot=0,ax = axes[1],title = 'Female',legend=False)
    #
    letter_prop = table/table.sum().astype(float)
    dny_ts = letter_prop.ix[['d','n','y'],'M'].T
    dny_ts.head()
    dny_ts.plot()

    all_names = top1000.name.unique()
    mask = np.array(['lesl' in x.lower() for x in all_names])
    lesley_like = all_names[mask]
    print lesley_like[:10]
    filtered = top1000[top1000.name.isin(lesley_like)]
    print filtered.groupby('name').births.sum()

    table = filtered.pivot_table('births',index='year',columns='sex',aggfunc='sum')
    table = table.div(table.sum(1),axis=0)
    table.tail()
    table.plot(style ={'M':'k-','F':'k--'})
    print 'over'
    # diversity.plot(title = 'Number of popular names in top 50%')



def add_prop(group):
    births = group.births.astype(float)
    group['prop'] = births/births.sum()
    return group
def get_top1000(group):
    return group.sort_values(by = 'births',ascending=False)[:1000]
def get_quantile_count(group, q=0.5):
    group = group.sort_values(by='prop',ascending=False)
    return group.prop.cumsum().searchsorted(q)+1


load_data()
