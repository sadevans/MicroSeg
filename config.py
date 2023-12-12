import os

Config = {}

Config['root_path'] = './'
Config['train_path'] = './data/train'
Config['test_path'] = './data/test'
Config['val_path'] = './data/val'
Config['batch_size'] = 2
Config['learning_rate'] = 0.001
Config['num_workers'] = 5
Config['num_epochs'] = 10

Config['train_dataset_name'] = 'E200_F-015_G42'
Config['train_dataset_path'] = os.path.join(Config['train_path'], Config['train_dataset_name'])

Config['train_dataset_imgs'] = os.path.join(Config['train_dataset_path'], 'denoised_scunet')
Config['train_dataset_masks'] = os.path.join(Config['train_dataset_path'], 'mask_scunet')
