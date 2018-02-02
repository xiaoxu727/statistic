import json
from collections import defaultdict
from pandas import DataFrame,Series
import pandas as pd
import numpy as np
def load_data():
    path = 'data/usagov_bitly_data2012-03-16-1331923249.txt'
    records = [json.loads(line) for line in open(path)]
    frame = DataFrame(records)
    return  frame
    # clean_tz = frame['tz'].fillna('Missing')
    # clean_tz[clean_tz ==''] ='Unknown'
    # tz_counts = clean_tz.value_counts()
    # # tz_counts = frame['tz'].value_counts()
    # # tz_counts[:10].plot(kind='barh',rot=0)
    # # print tz_counts[:10]
    # #
    # # time_zone = [rec['tz'] for rec in records if 'tz' in rec]
    # # return time_zone
    # return tz_counts

def get_counts(sequence):
    count = {}
    for x in sequence :
        if x in count:
            count[x] +=1
        else:
            count[x] =1
    return count

def get_counts2(sequence):
    counts = defaultdict(int)
    for x in sequence:
        counts[x] += 1
    return counts

def top_counts(count_dic, n = 2):
    value_key_pairs = [(count,tz) for tz,count in count_dic.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]

def cal_frame(frame):
    clean_tz = frame['tz'].fillna('Missing')
    clean_tz[clean_tz ==''] ='Unknown'
    tz_counts = clean_tz.value_counts()
    tz_counts[:10].plot(kine='barh',rot =0)

def contain(frame):
    cframe = frame[frame.a.notnull()]
    operating_system = np.where(cframe['a'].str.contains('Windows'),'Windows','Not Windows')
    # print operating_system
    by_tz_os = cframe.groupby(['tz',operating_system])
    agg_counts = by_tz_os.size().unstack().fillna(0)
    indexer = agg_counts.sum(1).argsort()
    count_subset = agg_counts.take(indexer)[-10:]
    normed_subset = count_subset.div(count_subset.sum(1),axis = 0 )

    normed_subset.plot(kind = 'barh',stacked = True)
    print normed_subset

frame = load_data()
# cal_frame(frame)
results = Series([x.split()[0] for x in frame.a.dropna()])
print results.value_counts()[:8]
contain(frame)

# counts = get_counts2(time_zone)
# print top_counts(counts,10)
# # print get_counts(time_zone)
# print get_counts2(time_zone)



