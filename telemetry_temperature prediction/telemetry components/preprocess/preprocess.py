import argparse
def preprocessing(data):
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn import preprocessing
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from datetime import datetime, timedelta
    from pandas import DataFrame
    from pandas import concat


    df = joblib.load(data)
    
    df.replace(['00:0f:00:70:91:0a', '1c:bf:ce:15:ec:4d', 'b8:27:eb:bf:9d:51'], [1, 2, 3], inplace=True)
    df['light'] = df['light']*1
    df['motion'] = df['motion']* 1

    # convert temperature from celsius to fahrenheit (°C to °F)
    df['temp'] = (df['temp'] * 1.8) + 32

    # convert unix time to time of day

    start = datetime(1970, 1, 1)  # Unix epoch start time
    df['datetime'] = df.ts.apply(lambda x: start + timedelta(seconds=x))

    data = df.set_index('datetime')

    data = data[['co','humidity', 'light', 'lpg', 'smoke', 'temp', 'motion']]
   
 
    def series_to_supervised(data,n_in=1,n_out=1,dropnan=True):
      n_vars = 1 if type(data) is list else data.shape[1]
      df = DataFrame(data)
      cols,names = list(),list()
      # input sequence(t-n,...t-1)
      for i in range(n_in,0,-1):
        cols.append(df.shift(i))
        names+= [('var%d(t-%d)'%(j+1,i))for j in range(n_vars)]
      #forecase sequence (t t+1,....,t+n)
      for i in range(0,n_out):
        cols.append(df.shift(-1))
        names = ['co', 'humidity', 'light', 'lpg', 'smoke', 'temp', 'motion', 'co1', 'humidity1', 'light1', 'lpg1', 'smoke1', 'temp1', 'motion1']
      #put it all together
      agg = concat(cols,axis=1)
      agg.columns = names
      #drop rows with nan values
      if dropnan:
        agg.dropna(inplace=True)
      return agg

    values = data.values
    data2= series_to_supervised(data)
    X= data2.drop(columns = ['temp1'])
    y = data2['temp1']
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=0)

    X_train= X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

  
    data_dic = {"X_train": X_train,"X_test": X_test, "Y_train": y_train, "Y_test": y_test}
    
    joblib.dump(data_dic, 'clean_data')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    args = parser.parse_args()
    preprocessing(args.data)