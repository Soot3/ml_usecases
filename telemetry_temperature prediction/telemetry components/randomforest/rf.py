import argparse
def rf(clean_data):
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn import metrics
    from sklearn.ensemble import RandomForestRegressor
    
    
    data = joblib.load(clean_data)

    X_train = data['X_train']
    y_train = data['Y_train']
    X_test = data['X_test']
    y_test = data['Y_test']

    rfreg = RandomForestRegressor(n_jobs=-1) 

    rfreg.fit(X_train, y_train)

    y_pred = rfreg.predict(X_test)

    # Test score
    test = rfreg.score(X_test, y_test)
    train = rfreg.score(X_train, y_train)
    print('test accuracy:')
    print(test)
    print('train accuracy:')
    print(train)



    rf_metrics = {'train':train, 'test':test, 'model':rfreg}
    joblib.dump(rf_metrics,'rf_metrics')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_data')
    args = parser.parse_args()
    rf(args.clean_data)