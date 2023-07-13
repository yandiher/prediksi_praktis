# TUMPUKAN REGRESSION
def tumpukan_regresi (x, y, scaler='minmax'):
    # import basic library
    import pandas as pd
    import time as t

    # SPLIT
    from sklearn.model_selection import train_test_split

    # SCALE
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler

    # REGRESSION
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import BaggingRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import RidgeCV
    from sklearn.linear_model import ElasticNet
    from sklearn.linear_model import ElasticNetCV
    from sklearn.linear_model import Lars
    from sklearn.linear_model import LarsCV
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import LassoCV
    from sklearn.linear_model import LassoLars
    from sklearn.linear_model import LassoLarsCV
    from sklearn.linear_model import ARDRegression
    from sklearn.linear_model import BayesianRidge
    from sklearn.linear_model import PoissonRegressor
    from sklearn.linear_model import TweedieRegressor
    from sklearn.linear_model import GammaRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neighbors import RadiusNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import LinearSVR #(need pipeline)
    from sklearn.svm import NuSVR #(need pipeline)
    from sklearn.svm import SVR #(need pipeline)
    from sklearn.tree import DecisionTreeRegressor

    # SCORE
    from sklearn.metrics import r2_score

    # split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42, stratify=y)
    
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
        AdaBoostRegressor(),
        BaggingRegressor(),
        ExtraTreesRegressor(),
        GradientBoostingRegressor(),
        RandomForestRegressor(),
        HistGradientBoostingRegressor(),
        GaussianProcessRegressor(),
        LinearRegression(),
        Ridge(),
        RidgeCV(cv=5),
        ElasticNet(),
        ElasticNetCV(cv=5),
        Lars(),
        LarsCV(cv=5),
        Lasso(),
        LassoCV(cv=5),
        LassoLars(),
        LassoLarsCV(cv=5),
        ARDRegression(),
        BayesianRidge(),
        PoissonRegressor(),
        TweedieRegressor(),
        GammaRegressor(),
        KNeighborsRegressor(),
        RadiusNeighborsRegressor(),
        MLPRegressor(max_iter=500),
        LinearSVR(),
        NuSVR(),
        SVR(),
        DecisionTreeRegressor()]

    # name the model
    names = [
        'AdaBoostRegressor',
        'BaggingRegressor',
        'ExtraTreesRegressor',
        'GradientBoostingRegressor',
        'RandomForestRegressor',
        'HistGradientBoostingRegressor',
        'GaussianProcessRegressor',
        'LinearRegression',
        'Ridge',
        'RidgeCV',
        'ElasticNet',
        'ElasticNetCV',
        'Lars',
        'LarsCV',
        'Lasso',
        'LassoCV',
        'LassoLars',
        'LassoLarsCV',
        'ARDRegression',
        'BayesianRidge',
        'PoissonRegressor',
        'TweedieRegressor',
        'GammaRegressor',
        'KNeighborsRegressor',
        'RadiusNeighborsRegressor',
        'MLPRegressor',
        'LinearSVR',
        'NuSVR',
        'SVR',
        'DecisionTreeRegressor']
    
    # make empty variabel. values will be added later
    scores = []
    times = []
    
    # make loop for every algorithm
    for m in models:
        starting = round(t.time())
        m.fit(x_train, y_train)
        y_pred = m.predict(x_test)
        score = r2_score(y_true=y_test, y_pred=y_pred)
        scores.append(score)
        ending = round(t.time())
        time_count = ending - starting
        time_counts = str(time_count) + 's'
        times.append(time_counts)
        
    # create dataframe
    result = pd.DataFrame({"Algorithm":names, 'Score':scores, 'Time':times})
    return result.sort_values(['Score'],ascending=False)
# code ends here

# TUMPUKAN KLASIFIKASI
def tumpukan_klasifikasi (x, y, scaler='minmax'):
    # import basic library
    import pandas as pd
    import time as t

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
    from sklearn.metrics import accuracy_score

    # split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42, stratify=y)
    
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
        MLPClassifier(max_iter=500),
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
    times = []
    
    # make loop for every algorithm
    for m in models:
        starting = round(t.time())
        m.fit(x_train, y_train)
        y_pred = m.predict(x_test)
        score = accuracy_score(y_true=y_test, y_pred=y_pred)
        scores.append(score)
        ending = round(t.time())
        time_count = ending - starting
        time_counts = str(time_count) + 's'
        times.append(time_counts)

        
    # create dataframe
    result = pd.DataFrame({"Algorithm":names, 'Score':scores, 'Time':times})
    return result.sort_values(['Score'],ascending=False)
# code ends here

# PROCESS DATA
def proses(x, y, scaler='minmax', save='no'):
    """
    return x_train, x_test, y_train, y_test
    you can also save the scale
    """
    # NECESSARY LIBRARY
    import pandas as pd
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    # SPLIT DATA
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42, stratify=y)

    # LOGIC
    if scaler == 'minmax':
        scale = MinMaxScaler()
    elif scaler == 'standard':
        scale = StandardScaler()
    else:
        pass

    # SCALE
    x_train = scale.fit_transform(x_train)
    x_test = scale.transform(x_test)
    
    # JOBLIB
    if save == 'yes':
        joblib.dump(scale, 'scale.joblib')
    elif save == 'show':
        return x_train, x_test, y_train, y_test

# code ends here

# REGRESI
def regresi(x_train, x_test, y_train, y_test, regressor='linear_regression', save='no'):
    """
    input: x_train, x_test, y_train, y_test, dan regressor
    return: score
    """
    # MODEL
    import joblib
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import BaggingRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import RidgeCV
    from sklearn.linear_model import ElasticNet
    from sklearn.linear_model import ElasticNetCV
    from sklearn.linear_model import Lars
    from sklearn.linear_model import LarsCV
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import LassoCV
    from sklearn.linear_model import LassoLars
    from sklearn.linear_model import LassoLarsCV
    from sklearn.linear_model import ARDRegression
    from sklearn.linear_model import BayesianRidge
    from sklearn.linear_model import PoissonRegressor
    from sklearn.linear_model import TweedieRegressor
    from sklearn.linear_model import GammaRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neighbors import RadiusNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import LinearSVR #(need pipeline)
    from sklearn.svm import NuSVR #(need pipeline)
    from sklearn.svm import SVR #(need pipeline)
    from sklearn.tree import DecisionTreeRegressor
    
    # SCORE
    from sklearn.metrics import r2_score
        
    # REGRESSOR
    regression = regressor
    if regression == 'linear_regression':
        reg = LinearRegression()
    elif regression == 'ada_boost_regression':
        reg = AdaBoostRegressor()
    elif regression == 'bagging_regression' :
        reg = BaggingRegressor()
    elif regression == 'extra_trees_regression' :
        reg = ExtraTreesRegressor()
    elif regression == 'gradient_boosting_regression' :
        reg = GradientBoostingRegressor()
    elif regression == 'random_forest_regression':
        reg = RandomForestRegressor()
    elif regression == 'hist_gradient_boost_regression':
        reg = HistGradientBoostingRegressor()
    elif regression == 'gaussian_process_regression':
        reg = GaussianProcessRegressor()
    elif regression == 'ridge':
        reg = Ridge()
    elif regression == 'ridge_cv':
        reg = RidgeCV()
    elif regression == 'elastic_net':
        reg = ElasticNet()
    elif regression == 'elastic_net_cv':
        reg = ElasticNetCV()
    elif regression == 'lars':
        reg = Lars()
    elif regression == 'lars_cv':
        reg = LarsCV()
    elif regression == 'lasso':
        reg = Lasso()
    elif regression == 'lasso_cv':
        reg = LassoCV()
    elif regression == 'lasso_lars':
        reg = LassoLars()
    elif regression == 'lasso_lars_cv':
        reg = LassoLarsCV()
    elif regression == 'ard_regression':
        reg = ARDRegression()
    elif regression == 'bayesian_ridge':
        reg = BayesianRidge()
    elif regression == 'poisson_regression':
        reg = PoissonRegressor()
    elif regression == 'tweedie_regression':
        reg = TweedieRegressor()
    elif regression == 'gamma_regression':
        reg = GammaRegressor()
    elif regression == 'kneighbors_regression':
        reg = KNeighborsRegressor()
    elif regression == 'radius_neighbor_regression':
        reg = RadiusNeighborsRegressor()
    elif regression == 'mlp_regression':
        reg = MLPRegressor()
    elif regression == 'linear_svr':
        reg = LinearSVR()
    elif regression == 'nu_svr':
        reg = NuSVR()
    elif regression == 'svr':
        reg = SVR()
    elif regression == 'decision_tree_regression':
        reg = DecisionTreeRegressor() 
    else:
        print('no regressor')

    # FIT DATA
    reg.fit(x_train, y_train)

    # PREDICT
    pred = reg.predict(x_test)

    # JOBLIB
    if save == 'yes':
        joblib.dump(reg, 'regressor.joblib')
# code ends here

# KLASIFIKASI
def klasifikasi(x_train, x_test, y_train, y_test, classifier='logistic_regression', save='no'):
    """
    input: x_train, x_test, y_train, y_test, dan classifier
    return: score
    """
    # MODEL
    import joblib
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
    from sklearn.metrics import accuracy_score
        
    # CLASSIFIER
    classification = classifier
    if classification == 'logistic_regression':
        clf = LogisticRegression()
    elif classification == 'ada_boost_classification':
        clf = AdaBoostClassifier()
    elif classification == 'bagging_classification':
        clf = BaggingClassifier()
    elif classification == 'extra_trees_classification':
        clf = ExtraTreesClassifier()
    elif classification == 'gradient_boost_classification':
        clf = GradientBoostingClassifier()
    elif classification == 'random_forest_classification':
        clf = RandomForestClassifier()
    elif classification == 'hist_gradient_boost_classification':
        clf = HistGradientBoostingClassifier()
    elif classification =='gausian_process_classification':
        clf = GaussianProcessClassifier()
    elif classification =='logistic_regression_cv':
        clf = LogisticRegressionCV()
    elif classification == 'passive_aggressive_classification':
        clf = PassiveAggressiveClassifier()
    elif classification == 'perceptron':
        clf = Perceptron()
    elif classification == 'ridge_classification':
        clf = RidgeClassifier()
    elif classification == 'ridge_cv_classification':
        clf = RidgeClassifierCV()
    elif classification == 'sgd_classification':
        clf = SGDClassifier()
    elif classification == 'sgd_one_svm':
        clf = SGDOneClassSVM()
    elif classification == 'bernoulli_nb':
        clf = BernoulliNB()
    elif classification == 'categorical_nb':
        clf = CategoricalNB()
    elif classification == 'complement_nb':
        clf = ComplementNB()
    elif classification == 'gaussian_nb':
        clf = GaussianNB()
    elif classification == 'multinomial_nb':
        clf = MultinomialNB()
    elif classification == 'kneighbors_classification':
        clf = KNeighborsClassifier()
    elif classification == 'radius_neighbors_classification':
        clf = RadiusNeighborsClassifier()
    elif classification == 'mlp_classification':
        clf = MLPClassifier()
    elif classification == 'linear_svc':
        clf = LinearSVC()
    elif classification == 'nu_svc':
        clf = NuSVC()
    elif classification == 'svc':
        clf = SVC()
    elif classification == 'decision_tree_classification':
        clf = DecisionTreeClassifier()
    else:
        print('no classifier')

    # FIT DATA
    clf.fit(x_train, y_train)

    # PREDICT
    pred = clf.predict(x_test)
    
    # RETURN
    return accuracy_score(y_test, pred)

    # JOBLIB
    if save == 'yes':
        joblib.dump(clf, 'regressor.joblib')
# code ends here


# nanti ditambah lagi