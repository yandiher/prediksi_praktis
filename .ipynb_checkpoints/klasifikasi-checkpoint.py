# FUNCTION FOR CLASSIFICATION
def tumpukan_klasifikasi (x, y, scaler='minmax'):
    # import basic library
    import pandas as pd

    # SPLIT
    from sklearn.model_selection import train_test_split

    # SCALE
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler

    # CLASSIFICATION
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.linear_model import PassiveAggressiveClassifier
    from sklearn.linear_model import Perceptron
    from sklearn.linear_model import RidgeClassifier
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.linear_model import SGDClassifier
    from sklearn.linear_model import SGDOneClassSVM
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.naive_bayes import CategoricalNB
    from sklearn.naive_bayes import ComplementNB
    from sklearn.naive_bayes import GaussianNB
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neighbors import RadiusNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import LinearSVC
    from sklearn.svm import NuSVC
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    # SCORE
    from sklearn.metrics import r2_score
    from sklearn.metrics import accuracy_score

    # split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
    
    # choosing the scaler
    if scaler == 'standard':
        scale = StandardScaler()
    elif scaler == 'minmax':
        scale = MinMaxScaler()
    
    # transform the x data
    x_train = scale.fit_transform(x_train)
    x_test = scale.transform(x_test)
    
    # list the model
    models = [
        AdaBoostClassifier(),
        BaggingClassifier(),
        ExtraTreesClassifier(),
        GradientBoostingClassifier(),
        RandomForestClassifier(),
        HistGradientBoostingClassifier(),
        LogisticRegression(max_iter=1000),
        PassiveAggressiveClassifier(),
        Perceptron(),
        RidgeClassifier(),
        SGDClassifier(),
        SGDOneClassSVM(),
        BernoulliNB(),
        CategoricalNB(),
        ComplementNB(),
        GaussianNB(),
        MultinomialNB(),
        KNeighborsClassifier(),
        RadiusNeighborsClassifier(),
        MLPClassifier(max_iter=1000),
        LinearSVC(),
        SVC(),
        DecisionTreeClassifier()]

    # name the model
    names = [
        'AdaBoostClassifier',
        'BaggingClassifier',
        'ExtraTreesClassifier',
        'GradientBoostingClassifier',
        'RandomForestClassifier',
        'HistGradientBoostingClassifier',
        'LogisticRegression',
        'PassiveAggressiveClassifier',
        'Perceptron',
        'RidgeClassifier',
        'SGDClassifier',
        'SGDOneClassSVM',
        'BernoulliNB',
        'CategoricalNB',
        'ComplementNB',
        'GaussianNB',
        'MultinomialNB',
        'KNeighborsClassifier',
        'RadiusNeighborsClassifier',
        'MLPClassifier',
        'LinearSVC',
        'SVC',
        'DecisionTreeClassifier']
    
    # make empty variabel. values will be added later
    scores = []
    
    # make loop for every algorithm
    for m in models:
        m.fit(x_train, y_train)
        y_pred = m.predict(x_test)
        score = accuracy_score(y_true=y_test, y_pred=y_pred)
        scores.append(score)
        
    # create dataframe
    result = pd.DataFrame({"Algorithm":names, 'Score':scores})
    return result.sort_values(['Score'],ascending=False)
# end here