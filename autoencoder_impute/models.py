import json
import os
import random
from random import randint

import pandas as pd
from keras import backend as K
from keras import regularizers
from keras.layers import Dense, ELU, Activation, BatchNormalization, Input, Lambda
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.optimizers import Adam

from .loss import rmse


class NNImpute:

    MAX = 1
    MIN = 0
    NA = 0

    def __init__(self, normalization, loss, metrics, activation, activation_kwargs, batch_normalization):
        """
        Creates instance of autoencoder imputer
        :param normalization: boolean of normalization is needed, MAX and MIN class attributes represent interval
                                of normalization
        :param loss: loss function for autoencoder training (from Keras, or from loss package)
        :param metrics: list of metrics for training, from Keras package
        :param activation: activation class object from Keras package
        :param activation_kwargs: constructor kwargs of activation class object
        :param batch_normalization: boolean if batch normalization is needed
        """
        self.model = None
        self._scaling_params = {}
        self._loss = loss
        self._metrics = metrics
        self._activation = activation
        self._activation_kwargs = activation_kwargs if activation_kwargs is not None else {}
        self._batch_normalization = batch_normalization
        self._normalization = normalization

    def save(self, name, path='.'):
        """
        Saves autoencoder imputation model
        :param name: name of the model to save
        :param path: path to save in
        """
        directory = os.path.join(path, name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.model.save_weights(os.path.join(directory, name + '.h5'))
        with open(os.path.join(directory, name + '.json'), 'w') as f:
            f.write(self.model.to_json())

        with open(os.path.join(directory, 'scaling_params.json'), 'w') as f:
            json.dump(self._scaling_params, f)

    def load(self, name, path='.'):
        """
        Loads autoencoder imputation model
        :param name: name of DAE
        :param path: path to load from
        """
        directory = os.path.join(path, name)

        with open(os.path.join(directory, name + '.json'), 'r') as f:
            self.model = model_from_json(f.read())
        self.model.load_weights(directory + '/' + name + '.h5')

        with open(os.path.join(directory, 'scaling_params.json'), 'r') as f:
            self._scaling_params = json.load(f)

        self.model.compile(loss=self._loss, optimizer=Adam(), metrics=self._metrics)

    def _normalize(self, data, train=True):
        """
        Normalize dataset per feature
        :param data: pandas Dataframe
        :param train: True if normalizing training dataset
        :return: normalized dataset
        """
        normalized = data.copy()

        for column in data.columns:
            if train:
                max_value = data[column].max()
                min_value = data[column].min()
                self._scaling_params[str(column)] = {
                    'max': float(max_value),
                    'min': float(min_value)
                }
            else:
                max_value = self._scaling_params[str(column)]['max']
                min_value = self._scaling_params[str(column)]['min']
            if max_value != min_value:
                a = (self.MAX - self.MIN) / (max_value - min_value)
                b = self.MAX - a * max_value
                normalized[column] = a * data[column] + b
            else:
                normalized[column] = 0

        return normalized

    def _denormalize(self, data):
        """
        Scale dataset back to inital feature intervals
        :param data: pandas Dataframe
        :return: denormalized dataset
        """
        for column in data.columns:
            max_value = self._scaling_params[str(column)]['max']
            min_value = self._scaling_params[str(column)]['min']
            if max_value != min_value:
                a = (max_value - min_value) / (self.MAX - self.MIN)
                b = max_value - a * self.MAX
                data[column] = a * data[column] + b
            else:
                data[column] = 0
        return data

    def _sample(self, data, corruption_factor, train=True):
        """
        Sample data for train or test
        :param data: pandas Dataframe
        :param corruption_factor: corruption factor of sampling
        :return: sampled dataset
        """
        if self._normalization:
            data_norm = self._normalize(data, train)
        else:
            data_norm = data
        data_train_norm = self._corrupt(data_norm, corruption_factor)

        return data_train_norm

    def _corrupt(self, data, corruption_factor):
        """
        Corrupt ( NAN class attribute ) given dataset with corruption factor
        :param data: pandas Dataframe
        :return: Noised dataset
        """
        samples = []
        columns = len(data.columns) * corruption_factor
        for i in range(1, int(columns)):
            x_noise = pd.concat([data.copy()])
            for index, row in x_noise.iterrows():
                random_columns = random.sample(list(x_noise.columns), randint(1, int(columns)))
                row[random_columns] = self.NA
            samples.append(x_noise)
        return pd.concat(samples), pd.concat([data for _ in range(len(samples))])

    def transform(self, x):
        """
        Reconstructs missing values in the dataset
        :param x: pandas Dataframe of data with missing values.
        :return: pandas Dataframe with reconstructed data
        """
        if self._normalization:
            x_norm = self._normalize(x, train=False)
        else:
            x_norm = x
        x_norm_filled = x_norm.fillna(self.NA)
        y_pred = pd.DataFrame(self.model.predict(x_norm_filled), columns=x.columns)
        x_norm[x_norm.isnull()] = y_pred
        return self._denormalize(x_norm) if self._normalization else x_norm

    def predict(self, x):
        """
        Returns complete DAE output
        :param x: pandas Dataframe of data with missing values.
        :return: pandas Dataframe raw output of DAE
        """
        if self._normalization:
            x_norm = self._normalize(x, train=False)
        else:
            x_norm = x
        y_pred = self.model.predict(x_norm)
        y_df = pd.DataFrame(y_pred, columns=x.columns)
        return self._denormalize(y_df) if self._normalization else y_df

    def evaluate(self, x, y):
        """
        Evaluates autoencoder model
        :param x: pandas Dataframe of data with missing values.
        :param y: pandas Dataframe of complete data.
        :return: evaluation score
        """
        if self._normalization:
            x_norm = self._normalize(x, train=False)
            y_norm = self._normalize(y, train=False)
        else:
            x_norm = x
            y_norm = x
        return self.model.evaluate(x_norm, y_norm)


class DaeImpute(NNImpute):

    def __init__(self, k, h, model_type='semi_regularized',
                 normalization=True, loss=rmse, metrics=None,
                 activation=ELU, activation_kwargs=None, batch_normalization=False,
                 l2_coef=0.0001):
        """
        Creates instance of denoising autoencoder imputer
        :param k: encoder/decoder layers count
        :param h: encode/decoder compress/unfold rate
        :param model_type: DAE architecture type, {regularized, semi_regularized}
        :param normalization: boolean of normalization is needed, MAX and MIN class attributes represent interval
                                of normalization
        :param loss: loss function for autoencoder training (from Keras, or from loss package)
        :param metrics: list of metrics for training, from Keras package
        :param activation: activation class object from Keras package
        :param activation_kwargs: constructor kwargs of activation class object
        :param batch_normalization: boolean if batch normalization is needed
        :param l2_coef: l2 regularization coefficient
        """
        super().__init__(loss=loss,
                         metrics=metrics,
                         activation=activation,
                         activation_kwargs=activation_kwargs,
                         batch_normalization=batch_normalization,
                         normalization=normalization)

        self._k = k
        self._h = h
        self._model_type = model_type

        self._l2_coef = l2_coef

    def fit(self, x, x_val=None, corruption_factor=0.5, epochs=500, batch_size=1000, callbacks=None, verbose=0):
        """
        Trains the model for a fixed number of epochs (iterations on a dataset).
        :param x: pandas Dataframe of training complete data.
        :param x_val: pandas Dataframe of validation complete data.
        :param corruption_factor: missing columns percentage
        :param epochs: Integer. Number of epochs to train the model.
        :param batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, it will default to 32.
        :param verbose: Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
        """
        if self.model is None:
            input_dim = x.shape[1]

            model = None
            if self._model_type == 'regularized':
                model = self._get_regularized_model(input_dim)
            elif self._model_type == 'semi_regularized':
                model = self._get_semi_regularized_model(input_dim)

            train = self._sample(x, corruption_factor, train=True)
            test = self._sample(x_val, corruption_factor, train=False)

            model.fit(train[0], train[1],
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=test,
                      callbacks=callbacks,
                      verbose=verbose)
            self.model = model
        else:
            train = self._sample(x, corruption_factor)
            test = self._sample(x_val, corruption_factor)
            self.model.fit(train[0], train[1],
                           epochs=epochs,
                           batch_size=batch_size,
                           validation_data=test,
                           callbacks=callbacks,
                           verbose=verbose)

    def _get_regularized_model(self, input_dim):
        """
        Constructs regularized DAE model
        :param input_dim: dataset features count, size of DAE input layer
        :return: compiled model
        """
        model = Sequential()
        model.add(Dense(input_dim + self._h, input_dim=input_dim, kernel_regularizer=regularizers.l2(self._l2_coef)))
        model.add(self._activation(**self._activation_kwargs))

        for i in range(2, self._k + 1):
            model.add(Dense(input_dim + i * self._h, kernel_regularizer=regularizers.l2(self._l2_coef)))

            if self._batch_normalization:
                model.add(BatchNormalization())

            model.add(self._activation(**self._activation_kwargs))

        model.add(Dense(input_dim + self._h * (self._k + 1), kernel_regularizer=regularizers.l2(self._l2_coef)))

        if self._batch_normalization:
            model.add(BatchNormalization())

        model.add(self._activation(**self._activation_kwargs))

        for i in range(self._k, 0, -1):
            model.add(Dense(input_dim + i * self._h, kernel_regularizer=regularizers.l2(self._l2_coef)))

            if self._batch_normalization:
                model.add(BatchNormalization())

            model.add(self._activation(**self._activation_kwargs))

        model.add(Dense(input_dim))
        model.add(Activation('linear'))

        model.compile(loss=self._loss, optimizer=Adam(), metrics=self._metrics)
        return model

    def _get_semi_regularized_model(self, input_dim):
        """
        Constructs semi regularized DAE model
        :param input_dim: dataset features count, size of DAE input layer
        :return: compiled model
        """
        model = Sequential()
        model.add(Dense(input_dim + self._h, input_dim=input_dim, kernel_regularizer=regularizers.l2(self._l2_coef)))
        model.add(self._activation(**self._activation_kwargs))

        for i in range(2, self._k + 1):
            model.add(Dense(input_dim + i * self._h, kernel_regularizer=regularizers.l2(self._l2_coef)))

            if self._batch_normalization:
                model.add(BatchNormalization())

            model.add(self._activation(**self._activation_kwargs))

        model.add(Dense(input_dim + self._h * (self._k + 1), kernel_regularizer=regularizers.l2(self._l2_coef)))

        if self._batch_normalization:
            model.add(BatchNormalization())

        model.add(self._activation(**self._activation_kwargs))

        for i in range(self._k, 0, -1):
            model.add(Dense(input_dim + i * self._h))

            if self._batch_normalization:
                model.add(BatchNormalization())

            model.add(self._activation(**self._activation_kwargs))

        model.add(Dense(input_dim))
        model.add(Activation('linear'))

        model.compile(loss=self._loss, optimizer=Adam(), metrics=self._metrics)
        return model


class MaeImpute(NNImpute):

    def __init__(self, k, h,
                 normalization=True, loss=rmse, metrics=None,
                 activation=ELU, activation_kwargs=None, batch_normalization=False,
                 l2_coef=0.0001):
        """
        Creates instance of multimodal autoencoder imputer
        :param k: encoder/decoder layers count
        :param h: encode/decoder compress/unfold rate
        :param normalization: boolean of normalization is needed, MAX and MIN class attributes represent interval
                                of normalization
        :param loss: loss function for autoencoder training (from Keras, or from loss package)
        :param metrics: list of metrics for training, from Keras package
        :param activation: activation class object from Keras package
        :param activation_kwargs: constructor kwargs of activation class object
        :param batch_normalization: boolean if batch normalization is needed
        :param l2_coef: l2 regularization coefficient
        """
        super().__init__(loss=loss,
                         metrics=metrics,
                         activation=activation,
                         activation_kwargs=activation_kwargs,
                         batch_normalization=batch_normalization,
                         normalization=normalization)

        self._k = k
        self._h = h
        self._l2_coef = l2_coef

    def fit(self, x, x_val=None, corruption_factor=0.5, epochs=500, batch_size=1000, callbacks=None, verbose=0):
        """
        Trains the model for a fixed number of epochs (iterations on a dataset).
        :param x: pandas Dataframe of training complete data.
        :param x_val: pandas Dataframe of validation complete data.
        :param corruption_factor: missing columns percentage
        :param epochs: Integer. Number of epochs to train the model.
        :param batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, it will default to 32.
        :param verbose: Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
        """
        if self.model is None:
            input_dim = x.shape[1]

            model = self._get_semi_regularized_model(input_dim)

            train = self._sample(x, corruption_factor, train=True)
            test = self._sample(x_val, corruption_factor, train=False)

            model.fit(train[0], train[1],
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=test,
                      callbacks=callbacks,
                      verbose=verbose)
            self.model = model
        else:
            train = self._sample(x, corruption_factor)
            test = self._sample(x_val, corruption_factor)
            self.model.fit(train[0], train[1],
                           epochs=epochs,
                           batch_size=batch_size,
                           validation_data=test,
                           callbacks=callbacks,
                           verbose=verbose)

    def _get_semi_regularized_model(self, input_dim):
        """
        Constructs semi regularized MAE model
        :param input_dim: dataset features count, size of DAE input layer
        :return: compiled model
        """
        model = Sequential()
        model.add(Dense(input_dim + self._h, input_dim=input_dim, kernel_regularizer=regularizers.l2(self._l2_coef)))
        model.add(self._activation(**self._activation_kwargs))

        for i in range(2, self._k + 1):
            model.add(Dense(input_dim + i * self._h, kernel_regularizer=regularizers.l2(self._l2_coef)))

            if self._batch_normalization:
                model.add(BatchNormalization())

            model.add(self._activation(**self._activation_kwargs))

        model.add(Dense(input_dim + self._h * (self._k + 1), kernel_regularizer=regularizers.l2(self._l2_coef)))

        if self._batch_normalization:
            model.add(BatchNormalization())

        model.add(self._activation(**self._activation_kwargs))

        for i in range(self._k, 0, -1):
            model.add(Dense(input_dim + i * self._h, kernel_regularizer=regularizers.l2(self._l2_coef)))

            if self._batch_normalization:
                model.add(BatchNormalization())

            model.add(self._activation(**self._activation_kwargs))

        model.add(Dense(input_dim))
        model.add(self._activation(**self._activation_kwargs))

        model.add(Dense(input_dim + self._h))
        model.add(self._activation(**self._activation_kwargs))

        for i in range(2, self._k + 1):
            model.add(Dense(input_dim + i * self._h))

            if self._batch_normalization:
                model.add(BatchNormalization())

            model.add(self._activation(**self._activation_kwargs))

        model.add(Dense(input_dim + self._h * (self._k + 1)))

        if self._batch_normalization:
            model.add(BatchNormalization())

        model.add(self._activation(**self._activation_kwargs))

        for i in range(self._k, 0, -1):
            model.add(Dense(input_dim + i * self._h))

            if self._batch_normalization:
                model.add(BatchNormalization())

            model.add(self._activation(**self._activation_kwargs))

        model.add(Dense(input_dim))
        model.add(Activation('linear'))

        model.compile(loss=self._loss, optimizer=Adam(), metrics=self._metrics)
        return model


class VaeImpute(NNImpute):

    def __init__(self, k, h, latent_dim=2,
                 normalization=True, loss=rmse, metrics=None,
                 activation=ELU, activation_kwargs=None, batch_normalization=False):
        """
        Creates instance of variational autoencoder imputer
        :param k: encoder/decoder layers count
        :param h: encode/decoder compress/unfold rate
        :param latent_dim: size of latent dimension
        :param normalization: boolean of normalization is needed, MAX and MIN class attributes represent interval
                                of normalization
        :param loss: loss function for autoencoder training (from Keras, or from loss package)
        :param metrics: list of metrics for training, from Keras package
        :param activation: activation class object from Keras package
        :param activation_kwargs: constructor kwargs of activation class object
        :param batch_normalization: boolean if batch normalization is needed
        """
        super().__init__(loss=loss,
                         metrics=metrics,
                         activation=activation,
                         activation_kwargs=activation_kwargs,
                         batch_normalization=batch_normalization,
                         normalization=normalization)

        self._k = k
        self._h = h
        self._latent_dim = latent_dim

    def fit(self, x, x_val=None, corruption_factor=0.5, epochs=500, batch_size=1000, callbacks=None, verbose=0):
        """
        Trains the model for a fixed number of epochs (iterations on a dataset).
        :param x: pandas Dataframe of training complete data.
        :param x_val: pandas Dataframe of validation complete data.
        :param corruption_factor: missing columns percentage
        :param epochs: Integer. Number of epochs to train the model.
        :param batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, it will default to 32.
        :param verbose: Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
        """
        if self.model is None:
            input_dim = x.shape[1]

            model = self._get_model(input_dim)
            train = self._sample(x, corruption_factor, train=True)
            test = self._sample(x_val, corruption_factor, train=False)
            model.fit(train[0], train[1],
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=test,
                      callbacks=callbacks,
                      verbose=verbose)
            self.model = model
        else:
            train = self._sample(x, corruption_factor)
            test = self._sample(x_val, corruption_factor)
            self.model.fit(train[0], train[1],
                           epochs=epochs,
                           batch_size=batch_size,
                           validation_data=test,
                           callbacks=callbacks,
                           verbose=verbose)

    def _get_model(self, input_dim):
        """
        Constructs VAE model
        :param input_dim: dataset features count, size of DAE input layer
        :return: compiled model
        """
        inputs = Input(shape=(input_dim,))

        last_layer = inputs

        for i in range(1, self._k + 1):
            last_layer = Dense(input_dim + i * self._h)(last_layer)
            if self._batch_normalization:
                last_layer = BatchNormalization()(last_layer)
            last_layer = self._activation(**self._activation_kwargs)(last_layer)

        z_mean = Dense(self._latent_dim, name='z_mean')(last_layer)
        z_log_var = Dense(self._latent_dim, name='z_log_var')(last_layer)

        z = Lambda(self._sampling, output_shape=(self._latent_dim,), name='z')([z_mean, z_log_var])

        last_layer = z
        for i in range(self._k, 0, -1):
            last_layer = Dense(input_dim + i * self._h)(last_layer)
            if self._batch_normalization:
                last_layer = BatchNormalization()(last_layer)
            last_layer = self._activation(**self._activation_kwargs)(last_layer)
        outputs = Dense(input_dim, activation='linear')(last_layer)
        vae = Model(inputs, outputs, name='vae')

        vae.compile(optimizer=Adam(), loss=self._vae_loss_gen(z_log_var, z_mean))
        return vae

    def _vae_loss_gen(self, z_log_var, z_mean):
        def _vae_loss(y_true, y_pred):
            reconstruction_loss = self._loss(y_true, y_pred)
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            return reconstruction_loss + kl_loss
        return _vae_loss

    @staticmethod
    def _sampling(args):
        """
        Reparameterization trick by sampling fr an isotropic unit Gaussian.
        :param args: mean and log of variance of Q(z|X)
        :return z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]

        epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=1)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon



