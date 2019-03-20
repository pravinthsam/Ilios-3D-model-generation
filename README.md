# Ilios-3D-model-generation
This project attempts to generate 3D voxel representations of object from a single 2D RGB segmented image. The project breaks down the problem into 2D to 2.5D (Depth map) prediction and 2.5D to 3D prediction. The project presentation slides with explanation and results can be viewed [here](http://bit.ly/pravinth3d).

## Getting Started

Clone this repo to a machine with an NVidia card (tested on aws p2.xlarge instance).

```
git clone https://github.com/pravinthsam/Ilios-3D-model-generation/
cd Ilios-3D-model-generation/
```

Create the environment file from the provided `environment.yaml` file

```
conda env create -f environment.yaml
source activate ilios
```

### Requirements

```
tensorflow=1.12.0
numpy=1.14.5
sklearn=0.19.1
skimage=0.13.1
scipy=1.1.0
boto3=1.9.82
matplotlib=2.2.2
```

## Gathering data

For this project the training was done on the `chair` category from the [shapenet](https://www.shapenet.org/) dataset. 2D and Depth map ground truth images were generated from the shapenet CAD models using blender. To install blender run the following commands.

```
wget https://download.blender.org/release/Blender2.78/blender-2.78-linux-glibc211-x86_64.tar.bz2
tar -xvf blender-2.78-linux-glibc211-x86_64.tar.bz2
echo export PATH=/home/ubuntu/blender-2.78-linux-glibc211-x86_64/:$PATH >> ~/.bashrc
source ~/.bashrc
```

The dataset is uploaded to a S3 bucket (shapenet-dataset). Running `data.py` pulls in the dataset from the bucket, renders the 2D and Depth images and also converts the CAD model to a voxel `.npy` numpy array file.

```
python ./src/data.py
```

Rendering and voxelization is a slow process (~2 minutes per CAD model), so please be patient.

## Training the two networks

### Download the trained model weights

If you wish to train the networks from scratch, you can follow the next subsections. Otherwise, the trained model weights are provided at [this google drive folder](https://drive.google.com/drive/folders/1CBcGWH9WIkvFMqCRCeAlK1y4WJedciHx?usp=sharing).
Place them under a folder called models. The folder structure should look like:

    .
    ├── src                     # Source files
    ├── models                  # Trained model weights
    │   ├── unet                # Depth-prediction network weights
    │   └── recgan              # Voxel-prediction network weights
    └── ...

### Training the Depth Prediction Network (U-Net 2D ConvNet)

The Depth prediction task takes in a 2D RGBA image (512x512x4) and converts it into a single channel 2D depth image (512x512x1). The model architecture chosen for this task is a U-Net with 4 levels and 3 2D conv layer at each level. A batch-norm operation is done at the output of each level.

To train/re-train the U-Net,
```
python ./src/model_unet.py
```

During training it was found that we got the minimum validation loss at ~20 epochs.

### Training the Voxel Prediction Network (3D-RecGan++ ConvNet GAN)

The Voxel prediction task takes in a 3D representation of the depth map (64x64x64x1) and converts it into an upscaled representation of the 3D model (256x256x256x1). The source for the architecture and initial model weights are from [this github project](https://github.com/Yang7879/3D-RecGAN-extended). The arxiv link for the author's work can be found [here](https://arxiv.org/abs/1802.00411).
Since the input to this model is the predicted depth map rather than the ground truth depth map, there was a need to retrain the model. The model was re-trained after loading the weights provided by the authors.

To train/re-train the 3D-RecGan,
```
python ./src/model_recgan.py
```

## Running Inference

To run only the inference, a `demo.py` file is provided. Since, inference needs to run 2 tensorflow networks in series, it was challenging to load and unload both models into the gpu (Refer to this issue [#19731](https://github.com/tensorflow/tensorflow/issues/19731)). Currently this was acheived using the `multiprocessing` module in python.

The input 2D images must be placed under `./demo/input` folder.

```
python ./src/demo.py
```

This will generate the depth images under `./demo/depth`. It will also generate the `.npy` voxel file and `.binvox` file at `./demo/voxel`.

## Viewing the final voxel output

To view the binvox files, you can use a program like viewvox which is a 3D model viewer. Viewvox can be downloaded [here](http://www.patrickmin.com/viewvox/).

```
./viewvox <path to binvox file>
```

The binvox file can also be viewed using `blender`.

# Acknowledgements

Thanks to [Bo Yang et al](https://github.com/Yang7879) for inspiring this project and providing the model and the code for the 3D-RecGAN architecture.




