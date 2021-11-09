def load_data():
    #importing libraries
    import joblib
    import pandas as pd
# importing the joblib files
    
    df1 = pd.read_csv('https://raw.githubusercontent.com/EkeminiUmanah/Uploads/main/Customer%20Propensity/data/training_sample1.csv')
    df2 = pd.read_csv('https://raw.githubusercontent.com/EkeminiUmanah/Uploads/main/Customer%20Propensity/data/training_sample2.csv')
    df3 = pd.read_csv('https://raw.githubusercontent.com/EkeminiUmanah/Uploads/main/Customer%20Propensity/data/testing_sample.csv')
    df = pd.concat([df1, df2, df3])
    
    
    joblib.dump(df, 'data')

if __name__ == '__main__':
    load_data()