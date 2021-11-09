import argparse
def valid(stats,schema):
  import tensorflow_data_validation as tfdv
  import joblib
  from tensorflow.python.lib.io import file_io
  import pandas as pd
  import json
  stats = joblib.load(stats)
  schema = joblib.load(schema)
  anomalies = tfdv.validate_statistics(statistics=stats, schema=schema)
  table=tfdv.utils.display_util.get_anomalies_dataframe(anomalies)
  if table.empty:
      metadata = {
        'outputs': [{
            'type': 'web-app',
            'storage': 'inline',
            'source': "<h4 style='color:green;'> No anomalies found.</h4>"
        }]
     }
  else:
      metadata = {
        'outputs': [{
            'type': 'table',
            'storage': 'inline',
            'format':'csv',
            'header': list(table.columns.values),
            'source': table.to_csv(header=False)
        }]
   }
  with file_io.FileIO('/mlpipeline-ui-metadata.json', 'w') as f:
      json.dump(metadata, f)
  tfdv.display_anomalies(anomalies)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--stats')
  parser.add_argument('--schema')
  args = parser.parse_args()
  valid(args.stats, args.schema)