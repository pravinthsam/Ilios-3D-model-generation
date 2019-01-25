import boto3
import os

def s3_list_subfolders(s3_bucket, s3_path):
    client = boto3.client('s3')
    result = client.list_objects(Bucket=s3_bucket,Delimiter='/', Prefix=s3_path)
    return [res.get('Prefix') for res in result.get('CommonPrefixes')]

def s3_download_folder(s3_bucket, s3_path, local_dest):
    client = boto3.client('s3')
    result = client.list_objects(Bucket=s3_bucket, Prefix=s3_path)
    for res in result.get('Contents'):
        #client.download_file(bucket, file.get('Key'), dest_pathname)
        dest_pathname = os.path.join(local_dest, res.get('Key'))
        if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
        client.download_file(s3_bucket, res.get('Key'), dest_pathname)
        print('Downloaded : {}'.format(res.get('Key')))

if __name__ == '__main__':

    # Test code which downloads 5 different models from the shapenet dataset
    for folder in s3_list_subfolders('shapenet-dataset', '03001627/')[:5]:
        print('Downloading from {}'.format(folder))
        s3_download_folder('shapenet-dataset', folder, 'data/raw')
        print()
