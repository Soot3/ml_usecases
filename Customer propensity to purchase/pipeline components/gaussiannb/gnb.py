import argparse
def gnb(clean_data):
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn import metrics
    from sklearn.naive_bayes import GaussianNB
    
    
    data = joblib.load(clean_data)
    X_train = data['X_train']
    y_train = data['Y_train']
    X_test = data['X_test']
    y_test = data['Y_test']

    gnb = GaussianNB() 

    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)

    # Test score
    test = gnb.score(X_test, y_test)
    train = gnb.score(X_train, y_train)
    print('test accuracy:')
    print(test)
    print('train accuracy:')
    print(train)

    #Classification Report
    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    print(df_classification_report)

    gnb_metrics = {'train':train, 'test':test, 'report':df_classification_report, 'model':gnb}
    joblib.dump(gnb_metrics,'gnb_metrics')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_data')
    args = parser.parse_args()
    gnb(args.clean_data)