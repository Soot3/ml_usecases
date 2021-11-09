import argparse
def cat(clean_data):
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn import metrics
    from catboost import CatBoostRegressor
    
    
    data = joblib.load(clean_data)

    X_train = data['X_train']
    y_train = data['Y_train']
    X_test = data['X_test']
    y_test = data['Y_test']


    cat = CatBoostRegressor()

    cat.fit(X_train,y_train)

    y_pred = cat.predict(X_test)



    # Test score
    test = cat.score(X_test, y_test)
    train = cat.score(X_train, y_train)
    print('test accuracy:')
    print(test)
    print('train accuracy:')
    print(train)


    cat_metrics = {'train':train, 'test':test, 'model':cat}
    joblib.dump(cat_metrics,'cat_metrics')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_data')
    args = parser.parse_args()
    cat(args.clean_data)