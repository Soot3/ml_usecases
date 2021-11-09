def load_data():
    #importing libraries
    import joblib
    import pandas as pd
    # importing the joblib files
    
    df = pd.read_csv('https://raw.githubusercontent.com/EkeminiUmanah/Uploads/f2b9c04f0e4fdb74b5a63afc33bdf813e02cd930/iot_telemetry_data.csv')
    
    
    joblib.dump(df, 'data')

if __name__ == '__main__':
    load_data()