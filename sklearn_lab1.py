########################## LinearRegression: file.csv #####################
import pandas as pd
df = pd.read_csv('file.csv',sep=';')
df.describe()

###############################
import matplotlib.pylab as plt
plt.scatter(df['alcohol'],df['quality'])
plt.xlabel('alcohol')
plt.ylabel('quality')
plt.title('alcohol-quality')
plt.show()

from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split

df = pd.read_csv('wine/winequality-red.csv', sep=';')
X = df[(list(df.columns))[:-1]]
y = df['quality']
X_tr, X_ts, y_tr, y_ts = train_test_split(X,y)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_predictions = regressor.predict(X_test)
regressor.score(X_test, y_test)
###############################
from sklearn.cross_validation import cross_val_score
# same as the above code
scores = cross_val_score(regressor, X, y, cv=5)
print(scores.mean(),scores)
########################### SGDRegressor: load_boston ########################
import numpy as np
from sklearn.datasets import load_boston
for sklearn.linear_model import SGDRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.preposessing import StandardScaler
from sklearn.cross_validation import train_test_split
data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
X_scaler = StandardScaler()
X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train)
X_test = X_scaler.transform(X_test)
y_test = y_scaler.transform(y_test)

regressor = SGDRegressor(loss='squared_loss')
scores = cross_val_score(regressor, X_train, y_train, cv=5)

######################## KNeightborsClassifier: load_iris #################
from sklearn import neightbors, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data,iris.target)

i_model = neightbors.KNeightborsClassifier(n_neighbors=1)
i_model.fit(X_train,y_train)

pre = i_model.predict(X_test)
print(accuracy_score(y_test,pre))

############################# load_iris ##################################
