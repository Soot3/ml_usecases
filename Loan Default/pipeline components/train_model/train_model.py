import argparse
def train(clean_data):
  import joblib
  from sklearn.ensemble import GradientBoostingClassifier

  data = joblib.load(clean_data)
  x_train = data['X_train']
  y_train = data['Y_train']

  gbc = GradientBoostingClassifier(n_estimators=1000, min_samples_split=5, max_depth=15)
  # Fitting model
  model = gbc.fit(x_train, y_train)
  joblib.dump(model,'fit_model')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--clean_data')
  args = parser.parse_args()
  train(args.clean_data)