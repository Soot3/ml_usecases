def load_data():
    #importing libraries
    import joblib
    import pandas as pd
    #importing the data
    data = pd.read_csv("https://raw.githubusercontent.com/Soot3/testing/master/loan_train1.csv")
    data1 = pd.read_csv("https://raw.githubusercontent.com/Soot3/testing/master/loan_train2.csv")
    df = pd.concat([data,data1])

    #serialize data to be used
    joblib.dump(df, 'data')

if __name__ == '__main__':
    load_data()