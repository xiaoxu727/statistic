from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import matplotlib.pyplot as plt

def read_file():
    data = pd.read_csv('data/Haiti.csv')
    print(data.info())
    return data


def to_cat_list(catstr):
    stripped = (x.strip() for x in str(catstr).split(','))
    return [x for x in stripped if x]


def get_all_categories(cat_series):
    cat_sets = (set(to_cat_list(x)) for x in cat_series)
    return sorted(set.union(*cat_sets))


def get_english(cat):
    code, names = cat.split('.')
    if '|' in names and not str.endswith(names, '|'):
        names = names.split('|')[1]
    return code, names.strip()


def get_code(seq):
    return [x.split('.')[0] for x in seq if x]


def proc_data():
    data = read_file()
    all_cats = get_all_categories(data.CATEGORY)
    english_mapping = dict(get_english(x) for x in all_cats if '.' in x)
    all_codes = get_code(all_cats)
    code_index = pd.Index(np.unique(all_codes))
    dummy_frame = DataFrame(np.zeros((len(data), len(code_index))), index=data.index, columns=code_index)
    for row, cat in zip(data.index, data.CATEGORY):
        codes = get_code(to_cat_list(cat))
        dummy_frame.ix[row, codes] = 1
    data = data.join(dummy_frame.add_prefix('category_'))
    print(data.info())
    # draw map
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    to_plot = ['2a', '1', '3c', '7a']

    lllat = 17.25
    urlat = 20.25
    lllon = -75
    urlon = -71
    for code, ax in zip(to_plot, axes.flat):
        m = basic_haiti_map(ax, lllat=lllat, urlat=urlat, lllon=lllon, urlon=urlon)
        cat_data = data[data['category_%s' % code] == 1]
        # cat_data = cat_data[cat_data['LONGITUDE'] != None and cat_data['LATITUDE'] != None ]
        # cat_data = cat_data.dropna(how='any')
        x, y = m(cat_data.LONGITUDE.values, cat_data.LATITUDE.values)
        m.plot(x, y, 'k.', alpha=0.5)
        ax.set_title('%s:%s' % (code, english_mapping[code]))
    plt.show()


def basic_haiti_map(ax=None, lllat=17.25, urlat=20.25,
                    lllon=-75, urlon=-71):
    m = Basemap(ax=ax, projection='stere', lon_0=(urlon+lllon)/2, lat_0=(urlat+lllat)/2,
                llcrnrlat=lllat, urcrnrlat=urlat, llcrnrlon=lllon, urcrnrlon=urlon, resolution='f')
    m.drawcoastlines()
    m.drawstates()
    m.drawcounties()
    return m


def add_layer():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    lllat = 17.25
    urlat = 20.25
    lllon = -75
    urlon = -71
    m = basic_haiti_map(ax, lllat=lllat, urlat=urlat, lllon=lllon, urlon=urlon)
    shapefile_path = 'data/PortAuPrince_Roads/PortAuPrince_Roads'
    m.readshapefile(shapefile_path, 'roads')
    plt.show()


def draw_map(data):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    to_plot = ['2a', '1', '3c', '7a']

    lllat = 17.25
    urlat = 20.25
    lllon = -75
    urlon = -71
    for code, ax in zip(to_plot, axes.flat):
        m = basic_haiti_map(ax, lllat=lllat, urlat=urlat, lllon=lllon, urlon=urlon)
        cat_data = data[data['category_%s' % code] == 1]
        x, y = m(cat_data.LONGITUDE, cat_data.LATITUDE)
        m.plot(x, y, 'k', alpha=0.5)
        # ax.set_tilte('%s:%s'%(code ))


def test():

    df = DataFrame(np.array(range(12)).reshape(3,4), index=['one', 'two', 'three'], columns=['A', 'B', 'C', 'D'])
    print(df)
    # 取一行
    print('df.ix[1]')
    print(df.ix[1])
    print('df.ix[:, 1:2]')
    print(df.ix[1:2, 1:2])
    # 取一行
    print('df.ix[two]')
    print(df.ix['two'])
    print('df.loc[two]')
    print(df.loc['two'])
    # 取列
    print("df[['A', 'B']]")
    print(df[['A', 'B']])
    # 索引列
    print('df.iloc[:, [1]]')
    print(df.iloc[:, [1, 2]])

    # 取元素
    print("df.loc['two']['A']")
    print(df.loc['two']['A'])
#     取块
    print(df.iloc[:1, [1, 2]])


if __name__ == '__main__':
    # data = read_file()
    # print(data[:5])
    # test()
    # proc_data()
    add_layer()