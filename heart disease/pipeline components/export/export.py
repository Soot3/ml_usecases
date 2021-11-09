import argparse
def export(bucket_name,credentials,best_model):

  import joblib
  from google.cloud import storage
  import requests
  from google.oauth2 import service_account
  response = requests.get(credentials)
  jsonResponse = response.json()
  joblib.dump(jsonResponse, 'model')
  # Explicitly use service account credentials by specifying the private key
  # file.
  credentials = service_account.Credentials.from_service_account_info(jsonResponse)
  storage_client = storage.Client(project='project_id', credentials=credentials)
  #storage_client = storage.Client.from_service_account_json('model')
  bucket=storage_client.get_bucket(bucket_name)

  blob = bucket.blob('heart/heart_model/best_model')
  blob.upload_from_filename(best_model)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--bucket_name')
  parser.add_argument('--credentials')
  parser.add_argument('--best_model')
  args = parser.parse_args()
  export(args.bucket_name, args.credentials,args.best_model)