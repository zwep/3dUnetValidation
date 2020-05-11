# encoding: utf-8

import re
import torch.utils.data
import collections
import nibabel as nib
import data_transforms as htransform
import numpy as np
import torch.utils.data
import os
import random

"""
Here we define generic components of the Data Generotr DataSet
"""


class DatasetGeneric(torch.utils.data.Dataset):
    """
    Generic Data Generic.
    """
    def __init__(self, ddata, input_shape, target_shape=None, shuffle=True, dataset_type='train', transform=None, file_ext='npy', **kwargs):
        kwargs = kwargs['kwargs']  # Needed because of... reasons
        # Variation options
        self.input_is_output = kwargs.get('input_is_output', False)  # Used to train a model as a decoder/encoder
        self.number_of_examples = kwargs.get('number_of_examples', None)  # Used to limit the number of samples
        self.selective_examples = kwargs.get('selective_examples', False)  # Used to select certain items
        self.masked = kwargs.get('masked', False)

        self.debug = kwargs.get('debug', None)
        self.ddata = ddata
        self.dataset_type = dataset_type

        self.input_dir, self.target_dir = self.set_load_dir()
        self.shuffle = shuffle

        # This will get all the input files...
        self.file_list = [x for x in os.listdir(self.input_dir) if x.endswith(file_ext)]
        if self.shuffle:
            random.shuffle(self.file_list)

        if self.selective_examples:
            self.file_list = [x for x in self.file_list if '__10' in x][:self.number_of_examples]
        else:
            self.file_list = self.file_list[:self.number_of_examples]

        # Check the image shape...
        self.img_input_shape = input_shape
        if target_shape is None:
            self.img_target_shape = input_shape
        else:
            self.img_target_shape = target_shape

        if self.debug:
            print('INFO - GEN: \t Total number of items ', len(self.file_list), '- Train: {}'.format(str(dataset_type)))
            self.__len__()
            print('INFO - GEN: \t Input path ', self.input_dir)
            print('INFO - GEN: \t Target path ', self.target_dir)
            print('INFO - GEN: \t Loaded image shape ', self.img_input_shape, self.img_target_shape)

    def __len__(self):
        n_files = len(self.file_list)
        if self.debug:
            print('INFO - GEN: \t Length of data generator ', n_files)
        return n_files

    def set_load_dir(self):
        if self.dataset_type == 'train':
            temp_dir = os.path.join(self.ddata, 'train')
        elif self.dataset_type == 'validation':
            temp_dir = os.path.join(self.ddata, 'validation')
        elif self.dataset_type == 'test':
            temp_dir = os.path.join(self.ddata, 'test')
        else:
            temp_dir = '~'
            print('Dataset unknown dataset type: ', self.dataset_type)

        input_dir = os.path.join(temp_dir, 'input')
        target_dir = os.path.join(temp_dir, 'target')

        if self.input_is_output:
            print('Target is equal to input')
            target_dir = input_dir

        if self.debug:
            print('INFO - GEN: \t Using image paths ', input_dir)
            print('INFO - GEN: \t                   ', target_dir)

        return input_dir, target_dir

    def on_epoch_end(self):
        'Updates file_list after each epoch'
        if self.shuffle:
            np.random.shuffle(self.file_list)

    def print(self):
        # Used to contain a fancy dict printer, but wanted to reduce the amount of code
        print(self.__dict__)


class UnetValidation(DatasetGeneric):
    def __init__(self, ddata, input_shape, target_shape=None, batch_perc=0.010, transform=None,
                 shuffle=True, dataset_type='train', file_ext='nii.gz', **kwargs):

        input_args = {k: v for k, v in locals().items() if (k !='self') and (k !='__class__')}
        super().__init__(**input_args)

        self.correct_elas_pixels = 10

        self.width = input_shape[0] + self.correct_elas_pixels  # This is the max...
        self.depth = input_shape[-1] + self.correct_elas_pixels  # This is the max as well... we can go lower..

        if target_shape is None:
            self.width_target = 44 + self.correct_elas_pixels  # This is different from input.. because of unpadded conv.
            self.depth_target = 28 + self.correct_elas_pixels  # This is the max as well... we can go lower..
        else:
            self.width_target = target_shape[0] + self.correct_elas_pixels
            self.depth_target = target_shape[-1] + self.correct_elas_pixels

        self.shift_center = np.array([self.width, self.width, self.depth])

    @staticmethod
    def _get_range(center, delta_x):
        temp_range = np.arange(center - delta_x // 2, center + delta_x // 2)
        return temp_range

    def __getitem__(self, index, x_center=None, y_center=None, t_center=None):
        """Generate one batch of data"""

        # Initialize the transforms....
        norm_trans = htransform.TransformNormalize(prob=False)  # Only X
        elas_trans = htransform.TransformElastic(mean=0, std=2, mode='constant')  # Part of the previous exercise
        unif_noise_trans = htransform.TransformUniformNoise()  # Only X
        bright_trans = htransform.TransformBrightness()  # Only X
        flip_trans = htransform.TransformFlip()

        if self.debug:
            print('EXEC - GEN: Transformers created')

        i_file = self.file_list[index]
        input_file = os.path.join(self.input_dir, i_file)
        if self.input_is_output:
            target_file = os.path.join(self.target_dir, i_file)
        else:
            target_file = os.path.join(self.target_dir, re.sub('image', 'mask', i_file))

        # Will be of shape 8, 8, X, Y
        x = nib.load(input_file).get_fdata()
        y = nib.load(target_file).get_fdata()

        if self.debug:
            print('EXEC - GEN: Loaded all data')

        n_x, n_y, n_z = x.shape
        if self.debug:
            print('INFO - GEN: loaded image shape ', x.shape)
            print('INFO - GEN: loaded target image shape ', y.shape)
            print('INFO - GEN: With index ', index, input_file, target_file)

        # # # For loop on data...
        x_center = np.random.randint(self.width // 2, n_x - self.width // 2 + 1)
        y_center = np.random.randint(self.width // 2, n_y - self.width // 2 + 1)
        t_center = np.random.randint(self.depth // 2, n_z - self.depth // 2 + 1)

        x_range_target = self._get_range(x_center, self.width_target)
        y_range_target = self._get_range(y_center, self.width_target)
        t_range_target = self._get_range(t_center, self.depth_target)

        num_target = 0
        while num_target == 0 and np.random.uniform(0, 1) > 0.05:
            x_center = np.random.randint(self.width // 2, n_x - self.width // 2 + 1)
            y_center = np.random.randint(self.width // 2, n_y - self.width // 2 + 1)
            t_center = np.random.randint(self.depth // 2, n_z - self.depth // 2 + 1)

            x_range_target = self._get_range(x_center, self.width_target)
            y_range_target = self._get_range(y_center, self.width_target)
            t_range_target = self._get_range(t_center, self.depth_target)

            y_check = np.take(y, x_range_target, axis=-3)
            y_check = np.take(y_check, y_range_target, axis=-2)
            y_check = np.take(y_check, t_range_target, axis=-1)
            num_target = y_check.sum()

        x_range = self._get_range(x_center, self.width)  # Extra pixel size to correct for elas deform
        y_range = self._get_range(y_center, self.width)
        t_range = self._get_range(t_center, self.depth)

        x = np.take(x, x_range, axis=-3)
        x = np.take(x, y_range, axis=-2)
        x = np.take(x, t_range, axis=-1)

        y = np.take(y, x_range_target, axis=-3)
        y = np.take(y, y_range_target, axis=-2)
        y = np.take(y, t_range_target, axis=-1)

        if self.debug:
            print('EXEC - GEN: subsampled image')
            print(f'\t shape x/y {x.shape}/{y.shape}')
            print('GEN UNET: counter unique values target', collections.Counter(y.ravel()))

        # BUT FIRST, let me apply some transformations
        x = norm_trans(x)
        if self.dataset_type == 'train':
            seed = np.random.randint(2147483647)  # make a seed with numpy generator
            np.random.seed(seed)  # Used so that the probs are all executed the same
            x = elas_trans(x)  # Part of the previous exercise
            x = flip_trans(x)
            x = norm_trans(x)
            x = unif_noise_trans(x)
            x = norm_trans(x)
            x = bright_trans(x)

            np.random.seed(seed)  # Used so that the probs are all executed the same
            y = elas_trans(y)  # Part of the previous exercise
            y = flip_trans(y)
            y = np.round(y)
        # y[y > 2] = 2 # Part of the previous exercise
        y[y > 1] = 1  # This is one of the drastic changes..
        y[y < 0] = 0  # I hope this is the answer

        if self.debug:
            print('EXEC - GEN: Transformations applied')
            print(f'\t shape x/y {x.shape}/{y.shape}')

        # Undo the correction for the elastic transform thingy
        x = x[self.correct_elas_pixels//2:-self.correct_elas_pixels//2,
              self.correct_elas_pixels//2:-self.correct_elas_pixels//2,
              self.correct_elas_pixels//2:-self.correct_elas_pixels//2]

        y = y[self.correct_elas_pixels//2:-self.correct_elas_pixels//2,
              self.correct_elas_pixels//2:-self.correct_elas_pixels//2,
              self.correct_elas_pixels//2:-self.correct_elas_pixels//2]

        # x = torch.cat(3 * [torch.as_tensor(x[np.newaxis])]) # Part of the previous exercise
        x = torch.as_tensor(x[np.newaxis])
        y = torch.as_tensor(y[np.newaxis].copy())

        if self.debug:
            print('EXEC - GEN: returning tensors')
            print('GEN UNET: counter unique values target', collections.Counter(y.numpy().ravel()))

        return x.float(), y.float()
