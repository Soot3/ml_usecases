import argparse
def statgen(data):
  import tensorflow_data_validation as tfdv
  import joblib
  from tensorflow.python.lib.io import file_io
  import json
  import pandas as pd

  df = joblib.load(data)
  stats = tfdv.generate_statistics_from_dataframe(df)
  html= tfdv.utils.display_util.get_statistics_html(stats) 

  metadata = {
        'outputs': [{
            'type': 'web-app',
            'storage': 'inline',
            'source': html
        }]
   }
  with file_io.FileIO('/mlpipeline-ui-metadata.json', 'w') as f:
       json.dump(metadata, f)

  
  joblib.dump(stats,'stats')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data')
  args = parser.parse_args()
  statgen(args.data)