# encoding: utf-8

import numpy as np

"""
Define a set of transforms over here... if needed...
"""

import scipy.signal
import scipy.ndimage as scndimage
import scipy.misc
import elasticdeform


class TransformElastic:
    def __init__(self, mean=0, std=4, grid_size=(32, 32, 32), mode='reflect', prob=True):
        self.displacement = np.random.normal(mean, std, size=(len(grid_size),) + grid_size)
        self.displacement[-1] = 0  # Setting the displacement in the z-axis to zero.
        self.prob = prob
        self.mode = mode

    def __call__(self, img):
        if self.prob:
            if np.random.uniform() >= 0.5:
                res = img
                return res

        if 'complex' == img.dtype:
            res_real = elasticdeform.deform_grid(img.real, displacement=self.displacement, mode=self.mode)
            res_img = elasticdeform.deform_grid(img.imag, displacement=self.displacement, mode=self.mode)
            res = res_real + 1j * res_img
        else:
            res = elasticdeform.deform_grid(img, displacement=self.displacement, mode=self.mode)

        return res


class TransformRotate:
    def __init__(self, angle_range=(-10, 10), axes=(-3, -2), prob=True):
        self.angle_range = angle_range

        self.axes = axes
        self.prob = prob

    def __call__(self, img):
        if self.prob:
            if np.random.uniform() >= 0.5:
                res = img
                return res

        angle = np.random.randint(*self.angle_range)
        res = scndimage.rotate(img, angle, axes=self.axes, reshape=False)

        return res


class TransformFlip:
    def __init__(self, axes=(-3, -2, -1), prob=True):
        self.axes = axes
        self.prob = prob

    def __call__(self, img):
        if self.prob:
            if np.random.uniform() >= 0.5:
                res = img
                return res

        ax = np.random.choice(self.axes, 1)
        res = np.flip(img, ax)

        return res


class TransformGaussianNoise:
    def __init__(self, mean=0, std=0.1, prob=True):
        self.mean = mean
        self.std = std
        self.prob = prob

    def __call__(self, img):
        if self.prob:
            if np.random.uniform() >= 0.5:
                res = img
                return res

        res = img + np.random.normal(self.mean, self.std, img.shape)

        return res


class TransformUniformNoise:
    def __init__(self, low=0, high=0.5, prob=True):
        self.low = low
        self.high = high
        self.prob = prob

    def __call__(self, img):
        if self.prob:
            if np.random.uniform() >= 0.5:
                res = img
                return res

        res = img + np.random.uniform(self.low, self.high, img.shape)

        return res


class TransformBrightness:
    def __init__(self, value=0.1, min_value=0, max_value=1, prob=True):
        self.value = value
        self.prob = prob
        self.min_value = min_value
        self.max_value = max_value

    @staticmethod
    def ceil_floor(img, min_value=0, max_value=1):
        img[img > max_value] = max_value
        img[img < min_value] = min_value
        return img

    def __call__(self, img):
        if self.prob:
            if np.random.uniform() >= 0.5:
                res = img
                return res

        random_value = np.random.uniform(0, self.value)
        res = self.ceil_floor(img + random_value, min_value=self.min_value, max_value=self.max_value)

        return res


class TransformStandardize:
    def __init__(self, prob=True):
        self.prob = prob

    def __call__(self, img, ax=None):
        if self.prob:
            if np.random.uniform() >= 0.5:
                res = img
                return res

        temp_mean = np.mean(img, axis=ax)
        temp_std = np.std(img, axis=ax)

        if isinstance(ax, tuple):
            new_shape = temp_mean.shape + (1,) * len(ax)  # Add new axes...
            temp_mean = temp_mean.reshape(new_shape)
            temp_std = temp_std.reshape(new_shape)
        elif isinstance(ax, int):
            new_shape = temp_mean.shape + (1,)  # Add one more ax
            temp_mean = temp_mean.reshape(new_shape)
            temp_std = temp_std.reshape(new_shape)
        elif ax is None:
            pass  # Dealing with onthing but scalars...
        else:
            print('errr')

        res = (img - temp_mean) / temp_std
        return res


class TransformNormalize:
    def __init__(self, min=0, max=1, prob=True):
        self.min = min
        self.max = max
        self.prob = prob

    def __call__(self, img):
        if self.prob:
            if np.random.uniform() >= 0.5:
                res = img
                return res

        res = (img - np.min(img)) * (self.max - self.min) / (np.max(img) - np.min(img)) + self.min

        return res


class TransformSmooth:
    def __init__(self, kernel_size, mode='same', prob=True):
        self.kernel = np.ones((kernel_size, kernel_size)) / kernel_size
        self.mode = mode
        self.prob = prob

    def __call__(self, img):
        if self.prob:
            if np.random.uniform() >= 0.5:
                res = img
                return res

        res = scipy.signal.convolve2d(img, self.kernel, mode=self.mode)

        return res
