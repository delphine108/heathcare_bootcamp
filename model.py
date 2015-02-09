from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.ensemble import GradientBoostingClassifier

def model(X_train, y_train, X_test):
    clf = Pipeline([('imputer', Imputer()),
                ('scaler', StandardScaler()),
                ('select', SelectPercentile(f_classif, 90)),
                ('clf', GradientBoostingClassifier(n_estimators=200, max_leaf_nodes=4, max_depth=None, random_state=2,
                   min_samples_split=5, subsample=0.5))])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)
    return y_pred, y_score
