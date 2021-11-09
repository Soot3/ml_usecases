import argparse
def results(metrics):
  import joblib
  data=joblib.load(metrics)
  print(f"Accuracy: \n {data['score']} \n")
  print(f"Confusion matrix: \n {data['cm']} \n")
  print(f"Classification report: \n {data['cr']} \n")
  with open('results.txt','w') as result:
    result.write(f"Accuracy: {data['score']} \n\n Confusion matrix: \n {data['cm']} \n\n Classification report: \n {data['cr']}")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--metrics')
  args = parser.parse_args()
  results(args.metrics)