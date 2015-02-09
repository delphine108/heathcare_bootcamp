from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn import svm

def model(X_train, y_train, X_test):
    weight={1.0:2, 0.0:1}
    clf = Pipeline([('imputer', Imputer()),
                ('scaler', StandardScaler()),
                ('select', SelectPercentile(f_classif, 90)),
                ('svm', svm.SVC(C=1.0, cache_size=200, class_weight=weight, coef0=0.0, degree=3,
                  gamma=0.0, kernel='rbf', max_iter=-1, probability=False, random_state=None,
                  shrinking=True, tol=0.001, verbose=False))])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)
    return y_pred, y_score
