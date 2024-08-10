from ultralytics import YOLO

# Load the model
model = YOLO('yolov10n.pt')

config_data = '/Users/mehmetbener/Desktop/Projects/Card Robot/input/Card Detector/train and export dataset/config.yaml'

parameters = {
    'data': config_data,
    'epochs': 20,
    'patience': 10,
    'batch': 16,
    'imgsz': 720,
    'save': True,
    'save_period': 1,
    'cache': True,
    'device': 'mps',
    'workers': 8,
    'project': '/Users/mehmetbener/Desktop/Projects/Card Robot/input/Card Detector',
    'name': '10 Class Test',
    'optimizer': 'SGD',
    'pretrained': True,
    'seed': 0,
    'rect': False,
    'cos_lr': True,
    'resume': False,
    'freeze': None,
    'lr0': 0.01,
    'lrf': 0.001,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'box': 10.0,
    'cls': 0.5,
    'dfl': 1.5,
    'label_smoothing': 0.1,
    'nbs': 64,
    'dropout': 0.0,
    'val': True,
    'plots': True
}

# Train the model
results = model.train(**parameters)
