from pandas import DataFrame, Series
import pandas as pd

############################## basic functions #################################
# basic functions: get partly info
data[['Director','id','Gerne','Runtime']]
data.ix[['Crazy Asian','Movie 2'],['Director','Runtime']]

# basic functions:math oprations
df1 = DataFrame(np.arange(12.).reshape((3,4)), columns=list('abcd'))
df2 = DataFrame(np.arange(20.).reshape(4,5), columns=list('abcde'))
df1-df2 # auto matching, fill NaN
df1.add(df2, fill_value=0)
df1.sub(df2, fill_value=0)
df1.mul(df2, fill_value=0)
df1.div(df2, fill_value=0)

df1.reindex(columns=df2.columns, fill_value=0)

# function map
df1 = DataFrame(np.random.randn(4,3), columns=list('abc'), index=['id1','id2','id3','id4'])
np.abs(df1)

func1 = lambda x: x.max()-x.min()
df1.apply(f, axis=1)

def f(x):
    return ([x.max(),x.min()],index=['max','min'])
df1.apply(f)

format11 = lambda x: '%.2f' % x
df1.applymap(format11)
df1['c'].map(format11)

# sorting
se1 = Series(np.arange(4),index=list('cadb'))
se1.sort_index()

df1 = DataFrame(np.arange(8).reshape((2,4)), index=['three','one'], columns=list('cdab'))
df1.sort_index(axis=1, ascending=False)

se2 = Series([-1, 6, np,nan, 3, -7])
se2.order()

df2 = DataFrame({'b':[1, 6, 3, -7], 'a':[2, 0, 4, -1]})
df2.sort_index(by='b')
df2.sort_index(by=['a','b']) # sort independently

se3 = Series([1, 6, 4, 3, 8, 7, 1])
se3.rank() # break same rank by 'arange avg ranking'
se3.rank(method='first')
se3.rank(method='min')
se3.rank(method='average')
se3.rank(method='max', ascending=False)

# 6. repeated index
se1.index.is_unique


############################# statistics #######################################
# summary
df1.sum()
df1.sum(axis=1) # aotu skip nan value
df1.mean(axis=1, skipna=False)
df1.idxmax()
df1.cumsum()
df1.describe()

se1 = Series(list('aabc')*4)
se1.describe()

# relative coefficients
import pandas.io.data as web
all_data = {}
start = '1/1/2007'
end = '1/1/2010'
for ticker in ['IBM', 'MSFT', 'GOOG']:
    all_data[ticker] = web.get_data_yahoo(ticker, start, end)

all_data['IBM'].head()

import pandas as pd
price = pd.DataFrame({tic:data['Adj Close'] for tic,data in all_data.items()})
volume = pd.DataFrame({tic:data['Volume'] for tic, data in all_data.items()})

returns = price.pct_change() # percentage_change of price
returns.tail()

returns.MSFT.corr(returns.GOOG) # compute the overlapped, not-nan corr coefficient
returns.MSFT.cov(returns.GOOG) # compute the covariance
returns.corr()
returns.cov()
returns.corrwith(returns.GOOG)
returns.corrwith(volume) # the correlation coefficient between pct_change and volume

# unique, isin, value_counts
se1 = pd.Series(list('cadaabbcc'))
a = se1.unique()
a.sort()

se1.value_counts()
pd.value_counts(se1, sort=False)

mask = se1.isin(['b','c'])
se1[mask]

data1 = pd.DataFrame({'Qu1':[1,3,4,3,4],'Qu2':[2,3,1,2,3], 'Qu3':[1,5,2,4,4]})
data1.apply(pd.value_counts).fillna(0)


############################## missing data ####################################
# missing data
se1 = pd.Series([None, 'aardvark','artichoke', np.nan, 'avocado'])
se1.isnull()
se1.dropna()
se1.fillna(0)

se1.dropna()
se1[se1.notnull()]

df1 = pd.DataFrame(np.arange(20.).reshape((4,5)))
df1[1][2:] = np.nan
df1.ix[3] = np.nan
df1[:][3] == df1[3][:] ==df[3] # they are the same
df1.dropna()
df1.dropna(how='all', axis=1)

# drop
df2 = pd.DataFrame(np.random.rand(7,3))
df2.loc[:4,1] = np.nan
df2.loc[:2,2] = np.nan
df2.dropna(thresh=3)

# fill
df2.fillna(0)
df2.fillna({1:0, 2:0.5}) # return new df3
_=df2.fillna({1:0,2:0.5}, inplace=True) # change df2
df2.fillna(method='ffill')
df2.fillna(method='ffill', limit=2)
df2.fillna(df2.mean())

########################## hierachical indexing ################################
data1 = pd.Series(np.random.rand(10),index=[list('aaabbbccdd'),[1,2,3]*2+[1,2,2,3]])
data1['b':'c']
data1.loc[['b','d']]
data1[:,2]
data1.unstack()
data1.unstack().stack()

data2 = pd.DataFrame(np.random.rand(5,3),index=[list('aabcb'),[1,2,1,2,2]],
columns=[['id','rate','runtime'],['d2','d1','d2']])
data2.index.names = ['genre', 'level' ]
data2.columns.names = ['feature', 'native' ]
MultiIndex.from_arrays([[],[]],name=[])######################
data2.swaplevel('genre', 'level')
data2.sort_index(level=1)
data2.sum(level='genre',axis=1)

df1 = pd.DataFrame({'a':range(7), 'b':range(7,0,-1), 'c':['one','one','one']+['two']*4, 'd':[0,1,2]*2+[3]})
df2 = df1.set_index(['c','d'])
df3 = df1.set_index(['c','d'], drop=False)
df2.reset_index()

########################## others ################################
# integer index
se1 = pd.Series(np.arange(3))
se2 = pd.Series(np.arange(3.), index=list('abc'))
se3 = pd.Series(np.arange(3.), index=[6,-1,3])
se3.iget_value(2)#?

# panel data structure
import pandas_datareader.data as web
start = '1/1/2007'
end = '1/1/2010'
pdata = pd.Panel(dict((stk,web.get_data_yahoo(stk, start,end)) for stk in ['IBM', 'MSFT', 'GOOG']))
