import boto3
import os
import glob
import subprocess
from sklearn.model_selection import train_test_split

def s3_list_subfolders(s3_bucket, s3_path):
    '''Returns a list of folders under a path inside a s3 bucket.'''

    client = boto3.client('s3')
    result = client.list_objects(Bucket=s3_bucket,Delimiter='/', Prefix=s3_path)
    return [res.get('Prefix') for res in result.get('CommonPrefixes')]

def s3_download_folder(s3_bucket, s3_path, local_dest):
    '''Downloads all contents inside a s3 path inside a bucket to a local folder. Does not work for root s3 path.'''
    client = boto3.client('s3')
    result = client.list_objects(Bucket=s3_bucket, Prefix=s3_path)
    for res in result.get('Contents'):
        #client.download_file(bucket, file.get('Key'), dest_pathname)
        dest_pathname = os.path.join(local_dest, res.get('Key'))
        if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
        client.download_file(s3_bucket, res.get('Key'), dest_pathname)
        print('Downloaded : {}'.format(res.get('Key')))


BLENDER_COMMAND = 'blender --background --python ./src/render_blender.py -- --views 50 --output_folder ./data/processed/{} {}'

def generate_depth_2d_images():
    '''Read all .obj files under data/raw and generate 2D and depth images'''
    # TODO add input for number of views and file paths
    for file_name in glob.glob('data/raw/**/*.obj', recursive=True):
        synset = file_name.split('/')[2]
        model_number = file_name.split('/')[3]

        print('Generating for model number : {}'.format(model_number))
        os.system(BLENDER_COMMAND.format(os.path.join(synset, model_number), file_name) + ' > /dev/null')

def train_test_filenames():
    '''Returns a list of file paths for training and testing'''
    # TODO move train_test_split to train.py
    output_files = glob.glob('./data/processed/**/*depth*.png', recursive=True)
    input_files = [f.replace('_depth', '').split('.')[1]+'.png' for f in output_files]

    return train_test_split(input_files, output_files, test_size=0.2)


if __name__ == '__main__':
    # TODO add argparse
    # Test code which downloads models from the shapenet dataset
    for folder in s3_list_subfolders('shapenet-dataset', '03001627/')[:1000]:
        print('Downloading from {}'.format(folder))
        s3_download_folder('shapenet-dataset', folder, 'data/raw')
        print()
    generate_depth_2d_images()
