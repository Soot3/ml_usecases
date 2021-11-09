import argparse
def train(clean_data):
  import joblib
  from catboost import CatBoostClassifier
  data = joblib.load(clean_data)
  x_train = data['X_train']
  y_train = data['Y_train']  
  cat = CatBoostClassifier(depth= 10, iterations= 100, l2_leaf_reg= 1, learning_rate= 0.1)
  model = cat.fit(x_train, y_train)
  joblib.dump(model,'fit_model')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--clean_data')
  args = parser.parse_args()
  train(args.clean_data)