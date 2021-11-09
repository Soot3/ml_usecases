import argparse
def preprocessing(data):
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn import preprocessing
    from imblearn.over_sampling import RandomOverSampler
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import RandomOverSampler
    
    df = joblib.load(data)
    
    
    #Split data into trin and test sets
    X = df.drop(['UserID', 'device_mobile', 'ordered', 'sign_in', ], axis=1)
    y = df['ordered']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    
    # oversample the smallest class of the train set
    over_sampler = RandomOverSampler(random_state=42)
    X_train, y_train = over_sampler.fit_resample(X_train, y_train)
    
    data_dic = {"X_train": X_train,"X_test": X_test, "Y_train": y_train, "Y_test": y_test}
    
    joblib.dump(data_dic, 'clean_data')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    args = parser.parse_args()
    preprocessing(args.data)