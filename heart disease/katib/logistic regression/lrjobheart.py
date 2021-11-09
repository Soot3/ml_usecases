import argparse
import os
from sklearn.linear_model import LogisticRegression as lr
import pandas as pd

#from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from timeit import default_timer as timestamp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_iter',
                        type = int,
                        default = 100,
                        help = 'The number of iterations for solvers to converge')
    parser.add_argument('--penalty',
                        type = str,
                        default = 'l2',
                        help = 'The norm used in penalization.')
    parser.add_argument('--solver',
                        type = str,
                        default = 'lbfgs',
                        help = 'Algorithm for optimization')
    args = parser.parse_args()
    #args = parser.parse_known_args()


    df=pd.read_csv('https://raw.githubusercontent.com/Soot3/testing/master/heart.csv')

    def removeOutlier(att, df):

      lowerbound = att.mean() - 3 * att.std()
      upperbound = att.mean() + 3 * att.std()

      df1 = df[(att > lowerbound) & (att < upperbound)]

      df = df1.copy()

      return df
    df = removeOutlier(df.trtbps, df)
    df = removeOutlier(df.chol, df)   

    # Separate Target Classes
    df_1 = df[df.output==1]
    df_2 = df[df.output==0]

    # Upsample minority class
    df_upsample_1 = resample(df_2, 
                                  replace=True,     # sample with replacement
                                  n_samples=163,    # to match majority class
                                  random_state=123) # reproducible results

    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_1, df_upsample_1])
    x = df_upsampled.drop('output', axis = 1)
    y = df_upsampled['output']  

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 111)
    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)



    params = {
        'max_iter': args.max_iter,
        'penalty': args.penalty,
        'solver': args.solver
    }

    #dtrain = rfc.DMatrix(X_train, y_train)
    #dtest = rfc.DMatrix(X_test, y_test)
    
    #dtrain = rfc.fit(X_train, y_train)
    #dtest = rfc.fit(X_test, y_test)
    
    

    start = timestamp()
    model = lr()
    model.fit(x_train, y_train)
    stop = timestamp()

    print('time=%.3f' % (stop - start))

    predictions = model.predict(x_test)

    print('accuracy=%.3f' % accuracy_score(y_test, predictions))
