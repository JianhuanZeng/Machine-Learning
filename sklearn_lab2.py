######################## KNeightborsClassifier: load_iris #################
from sklearn import neightbors, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X_tr, X_ts, y_tr, y_ts = train_test_split(iris.data,iris.target, random_state=0, train_size=0.8)

i_model = neightbors.KNeightborsClassifier(n_neighbors=3)
i_model.fit(X_tr,y_tr)

pre = i_model.predict(X_ts)
print(accuracy_score(y_ts,pre))

# cross_validation raw
i2_model_pre = i_model.fit(X_tr, y_tr).predict(X_ts)
i1_model_pre = i_model.fit(X_ts, y_ts).predict(X_tr)
accuracy_score(y1, i1_model_pre), accuracy_score(y2, i2_model_pre)

# cross_validation
from sklearn.cross_validation import cross_val_score
X = iris.data
y = iris.target
cross_val_score(i_model, X, y, cv=5)
############################# load_iris ##################################
