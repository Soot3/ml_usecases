import argparse
def eval(rf_metrics,keras_metrics,lr_metrics,sv_metrics, knn_metrics,cb_metrics):
  import joblib
  import pandas as pd
  import pprint
  rf_metrics=joblib.load(rf_metrics)
  keras_metrics=joblib.load(keras_metrics)
  lr_metrics=joblib.load(lr_metrics)
  sv_metrics=joblib.load(sv_metrics)
  knn_metrics=joblib.load(knn_metrics)
  cb_metrics=joblib.load(cb_metrics)
  lis = [rf_metrics,keras_metrics,lr_metrics,sv_metrics, knn_metrics,cb_metrics]
  max = 0
  for i in lis:
    accuracy = i['test']
    if accuracy > max:
      max = accuracy
      model = i['model']
      metrics = i
  
  print('Best metrics \n\n')
  pprint.pprint(metrics)

  joblib.dump(model, 'best_model')
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--rf_metrics')
  parser.add_argument('--keras_metrics')
  parser.add_argument('--lr_metrics')
  parser.add_argument('--sv_metrics')
  parser.add_argument('--knn_metrics')
  parser.add_argument('--cb_metrics')
  args = parser.parse_args()
  eval(args.rf_metrics,args.keras_metrics,args.lr_metrics,args.sv_metrics,args.knn_metrics,args.cb_metrics)