import numpy as np
import sklearn
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier

X, y = make_hastie_10_2(random_state=None)
X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]

#X_test[:,0] = 0
#X_test[:,0] = 1
#X_test[:,0] = 100000000
#n = y_test.size // 10
#X_test[:n,0] = np.nan
X_test[:,0] = np.nan

with sklearn.config_context(assume_finite=True): # disable nan valdiation 
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=None).fit(X_train, y_train)
    res = clf.score(X_test, y_test)
    print(res)
