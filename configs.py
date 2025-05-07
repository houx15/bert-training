data_output_dir_base = "data/training_data/"
output_dir_base = "model/"
log_dir_base = "logs/"

dataset_base = "data/original_data/"

default_regression_config = {'weight_decay': 0.3, 'warmup_steps': 200, 'learning_rate': 1e-05, 'num_train_epochs': 6, 'per_device_train_batch_size': 8}


default_binary_config = {'weight_decay': 0.3, 'warmup_steps': 100, 'learning_rate': 1e-05, 'num_train_epochs': 9, 'per_device_train_batch_size': 8}