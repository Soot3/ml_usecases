import argparse
import os
from sklearn.svm import SVC

import pandas as pd

#from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
#from timeit import default_timer as timestamp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--C',
                        type = float,
                        default = 1.0,
                        help = 'Regularization Parameter.')
    parser.add_argument('--kernel',
                        type = str,
                        default = 'rbf',
                        help = 'Specifies the kernel type to be used in the algorithm.')
    parser.add_argument('--max_iter',
                        type = int,
                        default = 1,
                        help = 'Hard limit on iterations within solver, or -1 for no limit.')
    args = parser.parse_args()
    
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
        'C': args.C,
        'kernel': args.kernel,
        'max_iter': args.max_iter
    }
    
    

    #start = timestamp()
    model = SVC()
    model.fit(x_train, y_train)
    #stop = timestamp()

    #print('time=%.3f' % (stop - start))

    predictions = model.predict(x_test)

    print('accuracy=%.3f' % accuracy_score(y_test, predictions))
