def load_data():
    #importing libraries
    from io import BytesIO
    import requests
    import joblib
    import pandas as pd
# importing the joblib files
    link = 'https://github.com/Soot3/testing/blob/master/los?raw=true'
    dfile = BytesIO(requests.get(link).content) 
    df = joblib.load(dfile)
    joblib.dump(df, 'data')

if __name__ == '__main__':
    load_data()