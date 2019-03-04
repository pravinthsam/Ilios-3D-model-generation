import boto3
import os
import glob
import subprocess
import binvox_rw
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import numpy as np

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
    for file_name in glob.glob('data/raw/*/*/models/*.obj'):
        synset = file_name.split('/')[2]
        model_number = file_name.split('/')[3]

        print('Generating depth for model number : {}'.format(model_number))
        os.system(BLENDER_COMMAND.format(os.path.join(synset, model_number), file_name) + ' > /dev/null')

def generate_3d_binvox(voxsize):
    for file_name in glob.glob('data/raw/*/*/models/*.obj'):
        synset = file_name.split('/')[2]
        model_number = file_name.split('/')[3]

        print('Generating for model number : {}'.format(model_number))
        print(file_name)
        with open(os.devnull, 'w') as devnull:
            cmd = "/home/ubuntu/binvox -d {0} -cb -e {1}".format(voxsize, file_name)
            ret = subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)
            if ret != 0:
                print("error", file_name)
            else:
                print(file_name)

def binvox_to_voxel(file_name):
    with open(file_name, 'rb') as f:
        voxel = binvox_rw.read_as_3d_array(f)
    return voxel.data

def generate_voxel_npy():
    for file_name in glob.glob('data/raw/*/*/models/*solid.binvox'):
        synset = file_name.split('/')[2]
        model_number = file_name.split('/')[3]

        print('Generating voxel for model number : {}'.format(model_number))
        voxel = binvox_to_voxel(file_name)
        # upsample to 256,256,256 since default binvox is 128,128,128
        voxel = resize(voxel, (256, 256, 256))>0
        np.save('data/processed/{}/{}/models/voxel.npy'.format(synset, model_number), (voxel*1).astype('uint8'))


def train_test_filenames():
    '''Returns a list of file paths for training and testing'''
    # TODO move train_test_split to train.py
    output_files = glob.glob('./data/processed/*/*/models/*depth*.png')
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
    generate_voxel_npy()
