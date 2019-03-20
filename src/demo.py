from model_unet import Network as Unet_Model
from model_recgan import Network as RecGan_Model
from config import config
import binvox_rw as binvox
import time
import glob
import numpy as np

from multiprocessing import Pool

def run_process(_process, process_name):
    '''
    Necessary because tensorflow allocates gpu memory
    till calling process ends
    '''
    start = time.time()
    p = Pool(1)
    p.map(_process, [()])
    p.close()
    p.join()
    end = time.time()
    print('{} took {}s...'.format(process_name, end-start))

def unet_inference(_):
    unet = Unet_Model(config)
    unet.build_graph()
    unet.demo()

def recgan_inference(_):
    recgan = RecGan_Model(config)
    recgan.build_graph()
    recgan.demo()

def pred_to_binvox(voxel_file):
    voxel = np.load(voxel_file)[:, :, :, 0]
    voxel = voxel > config['voxel_pred_threshold']
    bvox = binvox.Voxels(voxel, voxel.shape, [0.0, 0.0, 0.0], 0.684696, 'xyz')
    fname = voxel_file.replace('npy', 'binvox')
    with open(fname, 'wb') as f:
        bvox.write(f)

def binvox_generation(_):
    voxel_files = glob.glob('./demo/voxel/*.npy')
    for voxel_file in voxel_files:
        pred_to_binvox(voxel_file)

if __name__ == '__main__':

    # Depth inference
    run_process(unet_inference, 'Depth inference')

    # Voxel inference
    run_process(recgan_inference, 'Voxel inference')

    # Convert voxel npy files to binvox files
    run_process(binvox_generation, 'Binvox generation')
