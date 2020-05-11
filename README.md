Hi, thanks for looking into my code.

I reformatted everything so that you only need to read through this module. Hopefully that will make
the code-validation a bit easier.

Here I will explain what each file is intended to do

#### configuration_unet.json

I used this file in my main program to set all the details of my run. From data generator, to model choice and hyperparameters.
Under normal circumstances this file is created in a modular way.

#### data_generator.py

This contains the data generator setup for pytorch. Here I add all the data transformations and ways of loading the data

#### data_transforms.py

This contains classes to perform the data augmentation like elastic deforms, gaussian noise, etc.


#### model_definition.py

Here I define the 3D-unet version.


#### train_model.py

This is a slim version of my main training module. Here I load the configuration json-file to setup the model, data loaders and run parameters.
After loading the configuration, I can setup the data loader, loss, optimizer, and model object to perform the training.

I excluded the save_model method for now, and also the validate_model because they do not contribute to the model execution per-se. 
