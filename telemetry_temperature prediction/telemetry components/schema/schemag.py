import argparse
def schemagen(stats):
  import tensorflow_data_validation as tfdv
  import joblib
  import pandas as pd
  from tensorflow.python.lib.io import file_io
  import json
  stats = joblib.load(stats)
  schema = tfdv.infer_schema(stats)
  table=tfdv.utils.display_util.get_schema_dataframe(schema)
  
  metadata = {
        'outputs': [{
            'type': 'table',
            'storage': 'inline',
            'format':'csv',
            'header': ['Feature name','Type','Presence','Valency','Domain'],
            'source': table.to_csv(header=False)
        }]
   }
  with file_io.FileIO('/mlpipeline-ui-metadata.json', 'w') as f:
      json.dump(metadata, f)

  joblib.dump(schema,'schema')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--stats')
  args = parser.parse_args()
  schemagen(args.stats)