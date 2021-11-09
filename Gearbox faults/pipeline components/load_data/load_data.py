def load_data():
    #importing libraries
    import joblib
    import pandas as pd
    # importing the joblib files
    
    df = pd.read_csv('https://raw.githubusercontent.com/Soot3/testing/master/gearbox_data.csv')
    joblib.dump(df, 'data')

if __name__ == '__main__':
    load_data()
