import pandas as pd

############################## data reading #################################
# csv
pd.read_csv('filename.csv')
pd.read_table('filename.csv', sep=',')

# header
pd.read_csv('filename.csv', header=None)
pd.read_csv('filename.csv', names=['id', 'location', 'gender', 'age', 'race'])

# index_col
idx1 = ['id', 'location', 'gender', 'age', 'race']
pd.read_csv('filename.csv', names=idx1, index_col='id')
pd.read_csv('filename.csv', names=idx1, index_col=['id', 'location'])

# txt
pd.read_table('filename.txt', sep='/s+')

# skip rows
pd.read_csv('filename.csv', skiprows=[0,2,3])

# missing data
pd.read_csv('filename.csv', na_value=0)

sentinels={'age':[18,'NA'], 'race':['Black']}
pd.read_csv('filename.csv', na_value=sentinels)

# read partly
pd.read_csv('filename.csv', nrows=10)
chunker=pd.read_csv('filename.csv', chunker==1000)

tot = pd.Series([])
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(), fill_value=0)
tot = tot.order(ascending=False)

# data saving
tot.to_csv('filename.csv')
tot.to_csv('filename.csv', sep='|')
tot.to_csv('filename.csv', na_rep='NULL')
tot.to_csv(sys.stdout, index=False, header=False)
tot.to_csv(sys.stdout, index=False, col=list('abcd'))

dates = pd.date_range('1/1/2010', periods=7)
ts=pd.Series(np.arange(7),index=dates)
ts.to_csv('aug1.cvs')
pd.Series.from_csv('aug1.cvs', parse_datas=True)

# csv
import csv
f = open('aug15.cvs')

# json
import json
data = json.loads()

# web: html (hyper text makeup language)
from lxml.html import parse
from urllib2 import urlopen
p1 = parse(urlopen('http://finance.yahoo.com/q/op?s=AAPL+Options'))
p2 = p1.getroot()

links = p2.findall('.//a')
links[20].get('href')
links[20].text_content()

urls = [lnk.get('href') for lnk in p2.findall('.//a')]
urls[-10:]

tables = p2.findall('.//table')
table1 = tables[9]
rows = tables.findall('.//tr')

def _unpack(row, kind='td'):
    elts = row.findall('.//%s' % kind)
    return [vl.text_content() for vl in elts]
_unpack(rows[1],kind='th')

from pandas_datareader.parsers import TextParser
def parse_op_data(table):
    rows = table.findall('.//tr')
    header = _unpack(rows[1],kind='th')
    data = [ _unpack(r) for r in rows[1:]]
    return TextParser(data, names = header).get_chunk()
call_dt = parse_op_data(table1)

# web: xml (extensible makeup language)
from lxml import objectify
path = 'Performance_MNR.xml'
p3 = objectify.parse(open(path))
root = p3.getroot()

data = []
skips = ['PARENT_SEQ','INDICATOR_SEQ','DESIRED_CHANGE','DECIMAL_PLACES']
for elt in root.INDICATOR:
    el_data = {}
    for child in elt.getchildren():
        if child.tag in skips:
            continue
        el_data[child.tag] = child.pyval
    data.append(el_data)

pd.DataFrame(data)

from StringIO import StringIO
tag = ''
root = objectify.parse(StringIO(tag)).getroot()
root.get('href')
root.text
############################## binary data ####################################
# save data
df1 = pd.DataFrame(np.arange(20.).reshape((4,5)))
df1.save('df1_pickle')
pd.load('df1_pickle')

# hdf5: hierachical data format
Store = pd.HDFStore('mydata.h5')
Store['obj1'] = df1
Store['obj1_col'] =df1['col1']

# Excel, using xlrd and openpyxl packages
xls_files = pd.ExcelFile('mydata.xls')
table1 = xls_files.parse('Sheet1')

########################## HTML and Web API ################################
import requests
url = 'http://search.twitter.com/search.json?q=python%20pandas'
resp = requests.get(url)

import json
data = json.loads(resp.text)

inst_field = ['id', 'text', 'source']
tws = pd.DataFrame(data['result'], columns=inst_field)

########################## Database ################################
# SQLites
import sqlite3
query = '''
CREATE TABLE test (a )
'''
# MongoDB
