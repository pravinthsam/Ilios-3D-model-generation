from train import ShapesDataset, MEAN_DEPTH, STD_DEPTH
from model import UNet

import sys
from pathlib import Path
from tqdm import tqdm
from skimage import io
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms


if __name__ == '__main__':

    MODEL_WEIGHTS_PATH = './model_weights/unet.pth'
    model = UNet()

    if Path(MODEL_WEIGHTS_PATH).exists():
        print('Checkpoint found...')
        checkpoint = torch.load(MODEL_WEIGHTS_PATH)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print('Loaded model weights which have been trained for {} epochs'.format(start_epoch))


    if len(sys.argv) == 1:
        raise ValueError('There should be atleast one file name')

    input_files = sys.argv[1:]

    # TODO this part should be imported from load_dataset
    data_transform = transforms.Compose([transforms.ToTensor()])
    dataset = ShapesDataset(input_files, None, None)
    dataloader = DataLoader(dataset, batch_size=6,
                            shuffle=True, num_workers=4)
    #############

    DEVICE = 'cuda'

    model.to(DEVICE)

    all_outputs = []

    for d in tqdm(dataloader):
        inputs = d['input'].float().to(DEVICE)

        with torch.no_grad():
            outputs = model(inputs)
            mask = inputs[:, 3, :, :]
            mask = (mask<0.5)[:, None, :, :]
            outputs[mask] = 1.0
            outputs = outputs.to('cpu')
        outputs = outputs.detach().numpy()

        for i in range(outputs.shape[0]):
            depth = outputs[i, 0, :, :]
            depth = depth*STD_DEPTH + MEAN_DEPTH
            depth = np.clip(depth, 0.0, 255.0).astype('uint8')
            io.imsave('file{}.png'.format(i), depth)
