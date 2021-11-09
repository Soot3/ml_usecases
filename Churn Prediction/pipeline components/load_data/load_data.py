def load_data():
    #importing libraries
    import joblib
    import pandas as pd
    #importing the data
    data = pd.read_csv("https://raw.githubusercontent.com/Soot3/testing/master/cell2celltrain.csv",engine='python', encoding='utf-8', error_bad_lines=False)

    #serialize data to be used
    joblib.dump(data, 'data')

if __name__ == '__main__':
    load_data()