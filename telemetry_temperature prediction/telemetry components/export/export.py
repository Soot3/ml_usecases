import argparse
def export_model(bucket_name,best_model, minio_server, minio_access_key, minio_secret_key):
    import os
    import boto3
    import joblib
    import sklearn
    from botocore.client import Config


    minio_server='minio-service.kubeflow:9000'
    minio_access_key='minio'
    minio_secret_key='minio123'
    bucket_name='telemetry'
    best_model = joblib.load(best_model)

    s3 = boto3.resource('s3',
                    endpoint_url='http://minio-service.kubeflow:9000',
                    aws_access_key_id='minio',
                    aws_secret_access_key='minio123',
                    config=Config(signature_version='s3v4'),
                    region_name='us-east-1')


    # Create export bucket if it does not yet exist
    #response = s3.list_buckets()

    #s3.upload_file(model,
                #bucket_name,
               # ExtraArgs={"ACL": "public-read"},
           # )
           
    joblib.dump(best_model, 'best_model')
    
    s3.Bucket('telemetry').upload_file('best_model', 'best_model')
    #response = s3.list_objects(Bucket=telemetry)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket_name')
    parser.add_argument('--best_model')
    parser.add_argument('--minio_server')
    parser.add_argument('--minio_access_key')
    parser.add_argument('--minio_secret_key')
    args = parser.parse_args()
    export_model(args.bucket_name, args.best_model, args.minio_server, args.minio_access_key, args.minio_secret_key)