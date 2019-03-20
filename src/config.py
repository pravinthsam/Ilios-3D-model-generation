

config={}
config['batch_size'] = 4
config['vox_res_unet'] = 512
config['vox_res_x'] = 64
config['vox_res_y'] = 256
config['categories']=['03001627']
config['GPU'] = '0'
config['re_train'] = False
config['random_seed'] = 123
config['train_epochs'] = 1
config['bn_momentum'] = 0.95
config['learning_rate_unet'] = 0.01
config['MEAN_RGB'] = [0.0, 0.0, 0.0, 0.0]
config['STD_RGB'] = [1.0, 1.0, 1.0, 1.0]
config['voxel_pred_threshold'] = 0.65
