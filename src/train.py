from model_unet import UNet
from data import train_test_filenames
from skimage import io, transform
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import matplotlib.pyplot as plt
from pathlib import Path
import time
from tqdm import tqdm

MEAN_RGB = [92.15126579, 82.14264333, 73.88362667, 127.5]
STD_RGB = [108.91613243,  96.48436187,  85.80594293, 127.5]
MEAN_DEPTH = 0.0
STD_DEPTH = 255.0

class ShapesDataset(Dataset):
    '''Generates a Dataset given input 2D images and output 2.5D images'''

    def __init__(self, input_paths, output_paths, transform=None):

        self.input_paths = input_paths
        self.output_paths = output_paths
        self.transform = transform

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        input_path = './'+self.input_paths[idx]
        input_image = io.imread(input_path)
        input_image = (input_image-MEAN_RGB)/STD_RGB
        input_image = input_image.transpose((2, 0, 1))
        input_tensor = torch.from_numpy(input_image).double()

        if self.output_paths is  None:
            sample = {'input': input_tensor}
        else:
            output_path = './'+self.output_paths[idx]
            output_image = io.imread(output_path)[:, :, [0]]
            output_image = (output_image-MEAN_DEPTH)/STD_DEPTH
            output_image = output_image.transpose((2, 0, 1))
            output_tensor = torch.from_numpy(output_image).double()

            sample = {'input': input_tensor, 'output': output_tensor}

        return sample

def load_dataset():
    '''Generates the dataloader needed for batching iterator'''
    data_transform = transforms.Compose([transforms.ToTensor()])
    train_input, test_input, train_output, test_output = train_test_filenames()

    train_dataset = ShapesDataset(train_input, train_output, None)
    test_dataset = ShapesDataset(test_input, test_output, None)

    dataloader = DataLoader(train_dataset, batch_size=4,
                            shuffle=True, num_workers=4)
    return dataloader


def train_model(model, criterion, optimizer, scheduler, num_epochs, device, dataloader):
    '''Trains the model'''
    since = time.time()
    model = model.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        if scheduler is not None:
            scheduler.step()
        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for d in tqdm(dataloader):
            inputs = d['input'].float().to(device)
            depths = d['output'].float().to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)

                # Create mask to output so bg doesn't contribute to Loss
                mask = inputs[:, 3, :, :]
                mask = (mask<0.5)[:, None, :, :]
                outputs[mask] = 1.0

                loss = criterion(outputs, depths)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss/(len(dataloader)*dataloader.batch_size)

        print('Loss: {:.4f}'.format(epoch_loss))

        print()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return model

if __name__ == '__main__':

    NUM_EPOCHS = 1
    MODEL_WEIGHTS_PATH = './model_weights/unet.pth'

    dataloader = load_dataset()
    model = UNet()
    start_epoch = 0

    if Path(MODEL_WEIGHTS_PATH).exists():
        print('Checkpoint found...')
        checkpoint = torch.load(MODEL_WEIGHTS_PATH)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print('Loaded model weights which have been trained for {} epochs'.format(start_epoch))


    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=NUM_EPOCHS, device='cuda', dataloader=dataloader)
    torch.save({'epoch': start_epoch + NUM_EPOCHS, 'state_dict':model.state_dict()}, MODEL_WEIGHTS_PATH)
