import argparse
def schemagen(stats):
    import tensorflow_data_validation as tfdv
    import joblib
    import pandas as pd
    from tensorflow.python.lib.io import file_io
    import json
    stats = joblib.load(stats)
    schema = tfdv.infer_schema(statistics=stats)
    #tfdv.display(schema=schema)
    tfdv.utils.display_util.get_schema_dataframe(schema)
  

        
    joblib.dump(schema,'schema')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats')
    args = parser.parse_args()
    schemagen(args.stats)