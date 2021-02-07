import numpy as np
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, mains, appliances_regression, appliances_classification, window_size, batch_size, shuffle=False):
        self.mains = mains
        self.appliances_regression = appliances_regression
        self.appliances_classification = appliances_classification
        self.window_size = window_size
        self.batch_size = batch_size
        self.indices = np.arange(len(self.mains) - self.window_size + 1)
        self.shuffle = shuffle

    def __len__(self):
        return np.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, idx):
        mains_batch = []
        appliances_regression_batch = []
        appliances_classification_batch = []
        appliance_regression_sample = []
        appliance_classification_sample = []

        if idx == self.__len__() - 1:
            inds = self.indices[idx * self.batch_size:]
        else:
            inds = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]

        for i in inds:
            main_sample = self.mains[i:i + self.window_size]
            appliance_regression_sample = self.appliances_regression[i:i + self.window_size]
            appliance_classification_sample = self.appliances_classification[i:i + self.window_size]

            mains_batch.append(main_sample)
            appliances_regression_batch.append(appliance_regression_sample)
            appliances_classification_batch.append(appliance_classification_sample)

        mains_batch_np = np.array(mains_batch)
        mains_batch_np = np.reshape(mains_batch_np, (mains_batch_np.shape[0], mains_batch_np.shape[1], 1))
        appliances_regression_batch_np = np.array(appliances_regression_batch)
        appliances_regression_batch_np = np.reshape(appliances_regression_batch_np,
                                                    (appliances_regression_batch_np.shape[0],
                                                     appliances_regression_batch_np.shape[1]))
        appliances_classification_batch_np = np.array(appliances_classification_batch)
        appliances_classification_batch_np = np.reshape(appliances_classification_batch_np,
                                                        (appliances_classification_batch_np.shape[0],
                                                         appliances_classification_batch_np.shape[1]))
        return mains_batch_np, [appliances_regression_batch_np, appliances_classification_batch_np]

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.indices)
