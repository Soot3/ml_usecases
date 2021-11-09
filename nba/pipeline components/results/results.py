import argparse
def results(rf_metrics,gbc_metrics,lr_metrics):
  import joblib
  import pandas as pd
  import pprint
  rf_metrics=joblib.load(rf_metrics)

  gbc_metrics=joblib.load(gbc_metrics)
  lr_metrics=joblib.load(lr_metrics)

  print('Random Forest \n\n')
  pprint.pprint(rf_metrics)

  print('\n\n Gradient Boosting Classifier \n')
  pprint.pprint(gbc_metrics)
  print('\n\n Logistic regression \n')
  pprint.pprint(lr_metrics)

  with open('results.txt','w') as result:
    result.write(f"Random Forest: {rf_metrics} \n\n  Gradient Boosting Classifier: \n {gbc_metrics} \n\n Logistic regression: \n {lr_metrics} \n\n ")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--rf_metrics')

  parser.add_argument('--gbc_metrics')
  parser.add_argument('--lr_metrics')
  args = parser.parse_args()
  results(args.rf_metrics,args.gbc_metrics,args.lr_metrics)