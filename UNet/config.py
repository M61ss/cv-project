config = {
    "dataset": "OpenEDS-blur-augmented",
    'architecture': 'UNet',
    'input_channels': 1,
    'output_channels': 4,
    'channels': (4, 8, 16, 32),
    'strides': (2, 2, 2),
    'num_res_units': 2,
    'dropout': 0.5,
    'learning_rate': 1e-4,
    "epoches": 100
}