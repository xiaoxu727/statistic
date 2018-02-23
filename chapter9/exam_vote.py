import numpy as np
import pandas as pd
from pandas import DataFrame,Series
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
from matplotlib import  rcParams
from matplotlib.collections import LineCollection
import shapefile
# import dbflib
# import dbf

def load_data():
    fec = pd.read_csv('data/P00000001-ALL.csv')
    # print(fec.info())
    return fec

def unique_cands(fec):
    unique_cands = fec.cand_nm.unique()
    return unique_cands

def crate_parties():
    parties ={
    'Bachmann, Michelle': 'Republican',
    'Romney, Mitt': 'Republican',
    'Obama, Barack': 'Democrat',
    'Roemer, Charles E. Buddy III': 'Republican',
    'Pawlenty, Timothy': 'Republican',
    'Johnson, Gary Earl': 'Republican',
    'Paul, Ron': 'Republican',
    'Santorum, Rick': 'Republican',
    'Cain, Herman': 'Republican',
    'Gingrich, Newt': 'Republican',
    'McCotter, Thaddeus G': 'Republican',
    'Huntsman, Jon': 'Republican',
    'Perry, Rick': 'Republican'
    }
    return parties

def get_occ_mapping():
    occ_mapping = {
    'INFORMATION REQUESTED': 'NOT PROVIDED',
    'INFORMATION REQUESTED PER BEST EFFORTS ': 'NOT PROVIDED',
    'INFORMATION REQUESTED': 'NOT PROVIDED',
        'C.E.o.': 'CEO'
    }
    return occ_mapping


def get_emp_mapping():
    emp_mapping = {
        'INFORMATION REQUESTED PER BEST EFFORTS': 'NOT PROVIDED',
        'INFORMATION REQUESTED': 'NOT PROVIDED',
        'SELF': 'NOT PROVIDED',
        'SELF-EMPLOYED': 'NOT PROVIDED'
    }
    return emp_mapping


def add_party():
    fec = load_data()
    parties = crate_parties()
    fec['party']= fec.cand_nm.map(parties)
    # print(fec['party'].value_counts())
    return fec


def proc():
    fec = add_party()
    fec = fec[fec['contb_receipt_amt']> 0]
    fec_mrbo = fec[fec['cand_nm'].isin(['Romney, Mitt', 'Obama, Barack'])]
    # print(fec.contbr_occupation.value_counts()[:10])
    print(fec.contbr_employer.value_counts()[:10])
    occ_mapping = get_occ_mapping()
    f = lambda x: occ_mapping.get(x, x)
    fec.contbr_occupation = fec.contbr_occupation.map(f)
    emp_mapping = get_emp_mapping()
    f1 = lambda x:emp_mapping.get(x, x)
    fec.contbr_employer = fec.contbr_employer.map(f1)

    by_occupation = fec.pivot_table('contb_receipt_amt', index='contbr_occupation', columns='party', aggfunc='sum')
    over_2mmm = by_occupation[by_occupation.sum(1) > 2000000]
    print(over_2mmm)
    # over_2mmm.plot(kind='barh')
    # plt.show()
    grouped = fec_mrbo.groupby('cand_nm')
    print(grouped.apply(get_top_amounts, 'contbr_occupation', n=7))

    bins = np.array([0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000])
    labels = pd.cut(fec_mrbo.contb_receipt_amt, bins)
    print(labels)
    grouped = fec_mrbo.groupby(['cand_nm', labels])
    print(grouped.size().unstack(0))
    bucket_sums = grouped.contb_receipt_amt.sum().unstack(0)
    print(bucket_sums)
    normed_sums = bucket_sums.div(bucket_sums.sum(axis=1), axis=0)
    print(normed_sums)
    normed_sums[:-2].plot(kind='barh', stacked=True)
    # plt.show()
    grouped = fec_mrbo.groupby(['cand_nm', 'contbr_st'])
    totals = grouped.contb_receipt_amt.sum().unstack(0).fillna(0)
    totals = totals[totals.sum(1) > 100000]
    print(totals[:10])




def get_top_amounts(group, key, n=5):
    totals = group.groupby(key)['contb_receipt_amt'].sum()
    return totals.sort_values(ascending=False)[n:]

if __name__ == '__main__':
    # fec = load_data()
    # # print(fec.ix[123456])
    # for cand_nm in unique_cands(fec):
    #     print("'"+cand_nm+"':")

    # add_party()
    proc()