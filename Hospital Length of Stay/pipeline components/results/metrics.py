import argparse
def results(metrics):
  import joblib
  import pandas as pd
  data=joblib.load(metrics)
  print(f"Accuracy: \n {data['accuracy']} \n")
  print(f"Confusion matrix: \n {data['confusion']} \n")
  print(f"Precision: \n {data['precision']} \n")
  print(f"Recall: \n {data['recall']} \n")
  print(f"F1 score: \n {data['f1']} \n")
  with open('results.txt','w') as result:
    result.write(f"Accuracy: {data['accuracy']} \n\n Confusion matrix: \n {data['confusion']} \n\n Precision: \n {data['precision']} \n\n Recall: \n {data['recall']} \n\n F1 score: \n {data['f1']}")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--metrics')
  args = parser.parse_args()
  results(args.metrics)