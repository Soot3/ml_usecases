def load_data():
    #importing libraries
    from io import BytesIO
    import requests
    import joblib
    import pandas as pd
# importing the joblib files
    link = 'https://github.com/Soot3/testing/blob/master/vega_wrapper1?raw=true'
    dfile = BytesIO(requests.get(link).content) 
    df1 = joblib.load(dfile)

    link = 'https://github.com/Soot3/testing/blob/master/vega_wrapper2?raw=true'
    dfile = BytesIO(requests.get(link).content) 
    df2 = joblib.load(dfile)

    link = 'https://github.com/Soot3/testing/blob/master/vega_wrapper3?raw=true'
    dfile = BytesIO(requests.get(link).content) 
    df3 = joblib.load(dfile)

    link = 'https://github.com/Soot3/testing/blob/master/vega_wrapper4?raw=true'
    dfile = BytesIO(requests.get(link).content) 
    df4 = joblib.load(dfile)

    link = 'https://github.com/Soot3/testing/blob/master/vega_wrapper5?raw=true'
    dfile = BytesIO(requests.get(link).content) 
    df5 = joblib.load(dfile)

    link = 'https://github.com/Soot3/testing/blob/master/vega_wrapper6?raw=true'
    dfile = BytesIO(requests.get(link).content) 
    df6 = joblib.load(dfile)

    link = 'https://github.com/Soot3/testing/blob/master/vega_wrapper7?raw=true'
    dfile = BytesIO(requests.get(link).content) 
    df7 = joblib.load(dfile)

    df = pd.concat([df1,df2,df3,df4,df5,df6,df7])


    #serialize data to be used
    joblib.dump(df, 'data')

if __name__ == '__main__':
    load_data()