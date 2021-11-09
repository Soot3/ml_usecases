def load_data():

    #importing libraries
    from io import BytesIO
    import requests
    import joblib
    import pandas as pd
# importing the joblib files
    link = 'https://github.com/Soot3/testing/blob/master/listings_summary1?raw=true'
    dfile = BytesIO(requests.get(link).content) 
    df1 = joblib.load(dfile)

    link = 'https://github.com/Soot3/testing/blob/master/listings_summary2?raw=true'
    dfile = BytesIO(requests.get(link).content) 
    df2 = joblib.load(dfile)

    df = pd.concat([df1,df2])

    #serialize data to be used
    joblib.dump(df, 'data')

if __name__ == '__main__':
    load_data()
