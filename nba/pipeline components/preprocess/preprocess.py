import argparse
def preprocessing(data):
  import joblib
  import numpy as np
  import pandas as pd
  from sklearn.model_selection import train_test_split
  df = joblib.load(data)
  df.columns = df.columns.str.lower()
  df.dropna(axis=1, inplace =True)
  Q1 = df['touch_time'].quantile(0.25)
  Q3 = df['touch_time'].quantile(0.75)
  IQR = Q3 - Q1
  Lower_Whisker = Q1 -  1.5*IQR
  Upper_Whisker = Q3 + 1.5*IQR  
  outliers = df[(df.touch_time < Lower_Whisker) | (df.touch_time > Upper_Whisker)]
  df = df.drop(outliers.index)
  outliers = df[(df.touch_time < 0)]
  df = df.drop(outliers.index)
  df['location'] = np.where(df['location'] == 'H', 1, 0)
  df['w'] = np.where(df['w'] == 'W', 1, 0)
  df.shot_result = df.shot_result.map({"made":1,"missed":0})
  df.game_clock = df.game_clock.apply(lambda x: int(x.split(":")[0])*60 + int(x.split(":")[1]))
  df['date'] = df['matchup'].str[:12]
  df['date'] = pd.to_datetime(df['date'], format='%b %d, %Y')
  df = df.set_index('date')
  df = df.sort_index()
  df = df.drop(columns = ['player_id', 'player_name','matchup', 'game_id', 'closest_defender_player_id', 'closest_defender', 'fgm', 'pts'], axis=1)
  X  = df.drop('shot_result', axis=1)
  y = df['shot_result']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  data_dic = {"X_train": X_train,"X_test": X_test, "Y_train": y_train, "Y_test": y_test}

  joblib.dump(data_dic,'clean_data')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data')
  args = parser.parse_args()
  preprocessing(args.data)