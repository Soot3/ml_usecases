import argparse
def results(metrics,pca_metrics):
  import joblib
  import pandas as pd
  data=joblib.load(metrics)
  print("Autoencoder")
  if len(data) > 0:
    print(f"There are anomalies in the data, {len(data)} \n\n")
    print(data.head(20))
  else:
    print(f"There are no anomalies")
  print("\n\n **************** \n\n")
  data1=joblib.load(pca_metrics)
  print("PCA")
  if len(data1) > 0:
    print(f"There are anomalies in the data, {len(data1)} \n\n")
    print(data1.head(20))
  else:
    print(f"There are no anomalies")
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--metrics')
  parser.add_argument('--pca_metrics')
  args = parser.parse_args()
  results(args.metrics, args.pca_metrics)