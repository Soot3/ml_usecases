def load_data():
    #importing libraries
    import joblib
    import pandas as pd
# importing the joblib files
    
    df = pd.read_csv('https://raw.githubusercontent.com/charlesa101/KubeflowUseCases/draft/nba/dataset/shot_logs.csv?token=AMYXAKI4STPLOOS7Y6LCNWTAN5GNU')
    joblib.dump(df, 'data')

if __name__ == '__main__':
    load_data()