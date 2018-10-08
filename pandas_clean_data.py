##################################### I ####################################
###################### 1.pd.merge #######################
df1 = pd.DataFrame('key':list('bbacaab'),'data1':range(7))
df2 = pd.DataFrame('key':list('abd'), 'data2':range(3))
pd.merge(df1,df2)
pd.merge(df1,df2, on='key')
pd.merge(df1,df2, left_on='key1', right_on='key2')
pd.merge(df1,df2, how='outer')
pd.merge(df1,df2, on='key', how='left')
pd.merge(df1,df2, on='key', how='inner')
pd.merge(df1,df2, on='key', suffixes=('_key1','_key2'))
pd.merge(df1,df2, left_on='key1', right_index=True) # group value
# multi-indexing
pd.merge(df1,df2, left_on=['key1','key2'], right_index=True)
df1.join(df2, on='key', how='outer')
otherCol = pd.DataFrame(np.arange(17,18).reshape[4,2], index=list('acef'), columns=['NY', 'Oregon'])
df2.join([df1, otherCol], how='outer')

###################### 2.pd.concat #######################
# concatenation, stacking, binding
arr = np.arange(12).reshape((3,4))
np.concatenation([arr,arr],axis=1)

pd.concat([sr1,sr2,sr3])
pd.concat([sr1,sr2,sr3], axis=1) # become a dataframe

pd.concat([sr1,sr4], axis=1, join='inner')
pd.concat([sr1,sr4], axis=1, join_axes=[list('acbe'))

result=pd.concat([sr1,sr1,sr3],key=['one','two','thr'])
result.unstack()
pd.concat([sr1,sr2,sr3],axis=1,key=['one','two','thr'])

pd.concat([df1,df2],axis=1,key=['level1,level2'])
pd.concat({'level1':df1, 'level2':df2},axis=1)
pd.concat([df1,df2],axis=1,key=['level1,level2'],name=['upper','lower'])

pd.concat([df1,df2],axis=1,ignore_index=True)

###################### 3. combine_first for S#######################
a = pd.Series(np.nan, 2.5, np.nan, 3.5, 4.5, np.nan,index=list('fedcba'))
b = pd.Series(np.arange(len(a)),dtype=np.float64, index=list('fedcba')); b[-1]=np.nan
np.where(pd.isnull(a),b,a)

b[:-1].combine_first(a[1:])
df1.combine_first(df2)
################################### II #######################################
########################### 1. reshape; pivot #########################
data = pd.DataFrame(np.arange(6).reshape((2,3), columns=pd.Index(['one','two','three'],name='features'),index=pd.Index(['Ohio','Colorado'], name='state'))
result = data.stack() # transfer columns to rows
result.unstack(0) == result.unstack('state')

s1 = pd.Series([0,1,2,3], index=list('abcd'))
s2 = pd.Series([4,5,6], index=list('cde'))

data2 = pd.concat([s1,s2], key=['one','two'])
data2.unstack().stack(dropna=False)

######################### 2. pivot function for DF ####################
ldata=pd.DataFrame([], columns=['date','item','value'])
pivoted = ldata.pivot('date','item','value')
pivoted.head() # columns=different items, index_name=time

ldata=pd.DataFrame([], columns=['date','item','value','value2'])
pivoted = ldata.pivot('date','item')

###################################### III ####################################
########################### 1. duplicated data #########################
data.duplicated() # return Bealun
data.drop_duplicates()
data.drop_duplicates(['column1','column2'])
data.drop_duplicates(['column1','column2'], take_last=True)

################################# 2. map ##############################
data=pd.DataFrame([], columns=['food','ounces'])
meat2animal = {}
data['animal'] = data['food'].map(str.lower).map(meat2animal) #upper to lower
data['food'].map(lambda x: meat2animal[x.lower()])

########################### 3. replace #########################
data.replace(-9999,np.nan)
data.replace([-9999,-10000],np.nan)
data.replace([-9999,-10000],[np.nan, 0])
data.replace({-999:np.nan,-100000:0})

########################### 4. rename index #########################
########################### 1. reshape; pivot #########################

################################### IIII Strings ##############################
########################### 1. reshape; pivot #########################
########################### 1. reshape; pivot #########################
########################### 1. reshape; pivot #########################
########################### 1. reshape; pivot #########################
########################### 1. reshape; pivot #########################
