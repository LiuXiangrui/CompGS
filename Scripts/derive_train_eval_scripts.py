import os

trainer_path = r'..\Train.py'

gpcc_codec_path = r'C:\Users\XiangruiLIU\Desktop\mpeg-pcc-tmc13\build\tmc3\Release\tmc3.exe'  # path to GPCC codec

num_experiments = 5  # number of independent experiments for each configuration

dataset_configs = {
    'TanksAndTemplates': {
        'dataset_root':  r'D:\T2T\Datasets',
        'experiments_root': r'D:\T2T\Results3',
        'scene': {
            'Train': 'images',
            'Truck': 'images'
        },  # key is scene name, value is image folder name
        'config_path': r'..\Configs\TanksAndTemplates.yaml'
    },
    'DeepBlending': {
        'dataset_root': r'D:\DB\Dataset',
        'experiments_root': r'D:\DB\Results2',
        'scene': {
            'drjohnson': 'images',
            'playroom': 'images'
        },  # key is scene name, value is image folder name
        'config_path': r'..\Configs\DeepBlending.yaml'
    },
    'MipNeRF360': {
        'dataset_root':  r'D:\MipNeRF360\Dataset',
        'experiments_root': r'D:\MipNeRF360\Results3',
        'scene': {
            'bicycle': 'images_4',
            'flowers': 'images_4',
            'garden': 'images_4',
            'stump': 'images_4',
            'treehill': 'images_4',
            'room': 'images_2',
            'counter': 'images_2',
            'kitchen': 'images_2',
            'bonsai': 'images_2'
        },  # key is scene name, value is image folder name
        'config_path': r'..\Configs\MipNeRF360.yaml'
    }
}

model_configs = {
    'Lambda0_001': {
        'override_cfgs': {'gpcc_codec_path': gpcc_codec_path, 'lambda_weight': 0.001}},
    'Lambda0_005': {
        'override_cfgs': {'gpcc_codec_path': gpcc_codec_path, 'lambda_weight': 0.005}},
    'Lambda0_01': {
        'override_cfgs': {'gpcc_codec_path': gpcc_codec_path, 'lambda_weight': 0.01}},
}


if __name__ == '__main__':
    # generate command list
    cmd_list = []
    for dataset_name, dataset_config in dataset_configs.items():
        dataset_root, experiments_root = dataset_config['dataset_root'], dataset_config['experiments_root']
        scene_names = dataset_config['scene']
        config_path = dataset_config['config_path']

        os.makedirs(experiments_root, exist_ok=True)
        for model_name, model_config in model_configs.items():
            for scene_name, image_folder in scene_names.items():
                experiment_folder = os.path.join(experiments_root, scene_name)
                os.makedirs(experiment_folder, exist_ok=True)
                experiment_folder = os.path.join(experiment_folder, model_name)
                os.makedirs(experiment_folder, exist_ok=True)

                cmd = (f'python {trainer_path} --config {config_path} '
                       f'--root={os.path.join(dataset_root, scene_name)} --image_folder={image_folder} --save_directory={experiment_folder}')

                if 'override_cfgs' in model_config:
                    for key, value in model_config['override_cfgs'].items():
                        cmd += f' --{key}={value}'
                for _ in range(num_experiments):
                    cmd_list.append(cmd)

    # write commands to shell script
    if os.name == 'posix':  # linux
        with open('run.sh', 'w') as f:
            for cmd in cmd_list:
                f.write(f'{cmd}\n')
    else:  # windows
        with open('run.bat', 'w') as f:
            for cmd in cmd_list:
                f.write(f'{cmd}\n')
    print('Write commands to shell script successfully, please run it to start experiments.')