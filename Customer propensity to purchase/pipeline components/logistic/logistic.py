import argparse
def lr(clean_data):
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn import metrics
    from sklearn.linear_model import LogisticRegression
    
    data = joblib.load(clean_data)
    X_train = data['X_train']
    y_train = data['Y_train']
    X_test = data['X_test']
    y_test = data['Y_test']
        
    lr_model = LogisticRegression(max_iter=150, penalty='l2', C=1.0)
        
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
            
    # Test score
    test = lr_model.score(X_test, y_test)
    train = lr_model.score(X_train, y_train)
    print('test accuracy:')
    print(test)
    print('train accuracy:')
    print(train)
    
    #Classification Report
    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    print(df_classification_report)

    lr_metrics = {'train':train, 'test':test, 'report':df_classification_report, 'model':lr_model}
    joblib.dump(lr_metrics,'lr_metrics')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_data')
    args = parser.parse_args()
    lr(args.clean_data)        