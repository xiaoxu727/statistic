import pandas as pd

def load_data():
    unames = ['user_id','gender','age','occupation','zip']
    users = pd.read_table('data/movielens/users.dat',sep = '::',header = None, names = unames)
    rnames =['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_table('data/movielens/ratings.dat',sep='::', header= None, names = rnames)
    mnames = ['movie_id', 'title','genres']
    movies = pd.read_table('data/movielens/movies.dat', sep='::', header= None, names= mnames)

    data = pd.merge(pd.merge(ratings,users),movies)

    print data[:5]

    mean_rating =pd.pivot_table(data,index=['title'],columns=['gender'],values=['rating'],fill_value=0)
    # mean_rating =  data.pivot_table('rating',rows= 'title',cols='gender',aggfunc='mean')
    print mean_rating.info()

    # mean_rating = pd.pivot_table(data,index=['title'])
    # print mean_rating.info()
    ratings_by_tilte = data.groupby('title').size()
    active_titles = ratings_by_tilte.index[ratings_by_tilte>=250]
    mean_rating = mean_rating.ix(active_titles)

    print  mean_rating[:5]
    # top_female_rating = mean_rating.sort_index(by='F',ascending=False)
    # print top_female_rating[:10]
    # # print mean_rating.
    # mean_rating['diff'] = mean_rating['title']- mean_rating['title']
    # sorted_by_diff = mean_rating.sort_index(by ='diff')
    # print sorted_by_diff[:4]
    # print  data.info()
    rating_std_by_title = data.groupby('title')['rating'].std()
    rating_std_by_title = rating_std_by_title.ix[active_titles]
    print rating_std_by_title[:5]

load_data()
