# encoding: utf-8


import os
import json
import torch
import torch.utils.data
import numpy as np

config_file = 'configuration_unet.json'
template_path = os.path.dirname(config_file)

# Reads the json
# This contains a lot of information about my run.
with open(config_file, 'r') as f:
    text_obj = f.read()
    config_param = json.loads(text_obj)

# ddata stands for the directory where you have your data
# The data folder should be organised like
# data_directory
#       train
#           input
#           target
#       test (not necessary in this run)
#           input
#           target
#       validation (not necessary in this run)
#           input
#           target

config_param['dir']['ddata'] = "/home/bugger/Documents/data/grand_challenge/data"  # ==>
config_param['dir']['doutput'] = "/home/bugger/Documents/model_run/test_run"

# Create Unet
from model_definition import Unet3D
model_obj = Unet3D(**config_param['model']['config_unet3d'])

# Get a device
index_gpu = 0
device = torch.device("cuda:{}".format(index_gpu) if torch.cuda.is_available() else "cpu")
# Put model object to this device
model_obj.to(device)

# Create data generator and data loader
from data_generator import UnetValidation
data_generator_train = UnetValidation(dataset_type='train', **config_param['dir'], **config_param['data'])

data_loader_train = torch.utils.data.DataLoader(data_generator_train, batch_size=1, num_workers=0, shuffle=data_generator_train.shuffle)

# Train model
model_obj.train()

# get loss
loss_obj = torch.nn.BCEWithLogitsLoss()

# get optimizer
optimizer_obj = torch.optim.SGD(params=model_obj.parameters(), lr=0.0001)

# Start the training process. Set some initial values
min_epoch_loss = 9999
breakdown_counter = 0

epoch_loss_curve = []
val_loss_curve = []
history_dict = {}

n_epoch = config_param['model']['n_epoch']

# Parameters to control the stopping criteria based on the validation loss
breakdown_limit = config_param['callback']['breakdown_limit']
memory_length = config_param['callback'].get('memory_length', 5)
memory_time = config_param['callback'].get('memory_time', 5)

epoch = 0
while epoch < n_epoch and breakdown_counter < breakdown_limit:
    try:
        print(f"Epoch {epoch + 1}/{n_epoch} ...")
        print(f'Breakdown counter {breakdown_counter}')

        # Train
        epoch_loss = 0

        for container in data_loader_train:
            # Start with setting gradients to zero..
            optimizer_obj.zero_grad()

            X, y = container
            torch_input, torch_target = X.to(device), y.to(device)

            torch_pred = model_obj(torch_input)

            loss = loss_obj(torch_pred, torch_target)

            loss.backward()
            optimizer_obj.step()
            # batch_loss_curve.append(loss.item())
            epoch_loss += loss.item()


        # I ommit these statements to save some code lines
        # Normally this method contains statements like
        #   model_obj.eval()
        #   ~ perform validation
        #   model_obj.train()

        # val_loss = validate_model()
        # val_loss_curve.append(val_loss)

        # Average over amount of batches..
        epoch_loss = epoch_loss / data_loader_train.__len__()
        epoch_loss_curve.append(epoch_loss)
        print('average loss over batch: ', epoch_loss)

        # Check if we have a new minimum.
        # Store the weights for that state
        if epoch_loss < min_epoch_loss:
            min_epoch_loss = epoch_loss
            temp_weights = model_obj.state_dict()

        # Check if we have a proper less.. then, check if have enough decrease to continue
        # It looks like a quite elaborate way to check if we are improving or not.
        # I know that this acts as a regulizer as well, since it can stop the training 'too' soon.
        # I cant say if this is the cause for the bad performance on the model.
        if epoch_loss != 0:
            # For all the non-zero losses...
            temp_curve = [x for x in val_loss_curve if x != 0]
            historic_loss = np.mean(temp_curve[-(memory_time + memory_length): -memory_time])
            current_loss = np.mean(temp_curve[-memory_length:])
            criterion = historic_loss - current_loss

            # If the current loss gets larger than the historic loss...
            if criterion < 0:
                breakdown_counter += 1
            else:
                breakdown_counter -= 1
                breakdown_counter = max(breakdown_counter, 0)

        epoch += 1

        # Every now and then.. save intermediate results..
        if epoch % max(int(0.10 * n_epoch), 1) == 0:
            dir_temp_weights = os.path.join(config_param['dir']['doutput'], 'temp_weights.pt')
            torch.save(temp_weights, dir_temp_weights)
            # plt.close('all')
            # Here I would save intermediate model outputs for me to check during training time.
            # save_model(plot_name='intermediate')

    except KeyboardInterrupt:
        print('\t\t Keyboard interrupt ')
        break

if breakdown_counter > breakdown_limit:
    print('INFO - EXEC: \t We are not advancing fast enough. Broke out of training loop')
elif epoch >= n_epoch:
    print('INFO - EXEC: \t Completed all epochs')
else:
    print('INFO - EXEC: \t Increase in loss.. break down')

history_dict['train_loss'] = epoch_loss_curve
history_dict['val_loss'] = val_loss_curve
