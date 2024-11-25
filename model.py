import os

from keras import Model

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf
import numpy as np
import sys

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model


class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model = self._build_model(num_layers, width)


    def _build_model(self, num_layers, width):
        """
        Build and compile a fully connected deep neural network
        """
        inputs = keras.Input(shape=(self._input_dim,))
        x = layers.Dense(width, activation='relu')(inputs)
        for _ in range(num_layers):
            x = layers.Dense(width, activation='relu')(x)
        outputs = layers.Dense(self._output_dim, activation='linear')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='my_model')
        model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self._learning_rate))
        return model
    

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state, verbose=0)


    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        return self._model.predict(states, verbose=0)


    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """
        self._model.fit(states, q_sa, epochs=1, verbose=0)


    def save_model(self, path):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        self._model.save(os.path.join(path, 'trained_model.h5'))
        # plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True, show_layer_names=True)


    @property
    def input_dim(self):
        return self._input_dim


    @property
    def output_dim(self):
        return self._output_dim


    @property
    def batch_size(self):
        return self._batch_size


class TestModel:
    def __init__(self, input_dim, model_path):
        self._input_dim = input_dim
        self._model = self._load_my_model(model_path)


    def _load_my_model(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = os.path.join(model_folder_path, 'trained_model.h5')
        
        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            return loaded_model
        else:
            sys.exit("Model number not found")


    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)


    @property
    def input_dim(self):
        return self._input_dim

# class ActorModel:
#     def __init__(self, num_layers, width, learning_rate, input_dim, output_dim, batch_size):
#         self._input_dim = input_dim
#         self._output_dim = output_dim
#         self._learning_rate = learning_rate
#         self._batch_size = batch_size
#         self._model = self._build_model(num_layers, width)
#         self._optimizer = Adam(lr=self._learning_rate)
#
#     def _build_model(self, num_layers, width):
#         inputs = keras.Input(shape=(self._input_dim,))
#         x = layers.Dense(width, activation='relu')(inputs)
#         for _ in range(num_layers):
#             x = layers.Dense(width, activation='relu')(x)
#         outputs = layers.Dense(self._output_dim, activation='softmax')(x)
#         model = keras.Model(inputs=inputs, outputs=outputs, name='actor_model')
#         return model
#
#     def predict(self, state):
#         state = np.reshape(state, [1, self._input_dim])
#         return self._model.predict(state, verbose=0)
#
#     def train(self, states, actions, advantages):
#         actions = np.array(actions)
#         advantages = np.array(advantages)
#         with tf.GradientTape() as tape:
#             probs = self._model(states, training=True)
#             action_masks = tf.one_hot(actions, self._output_dim)
#             log_probs = tf.math.log(tf.reduce_sum(probs * action_masks, axis=1) + 1e-10)
#             loss = -tf.reduce_mean(log_probs * advantages)
#         # grads = tape.gradient(loss, self._model.trainable_variables)
#         # self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
#         # In ActorModel's train method
#         grads = tape.gradient(loss, self._model.trainable_variables)
#         clipped_grads = [tf.clip_by_norm(g, 1.0) for g in grads]
#         self._optimizer.apply_gradients(zip(clipped_grads, self._model.trainable_variables))
#
#     @property
#     def batch_size(self):
#         return self._batch_size
#
# class CriticModel:
#     def __init__(self, num_layers, width, learning_rate, input_dim):
#         self._input_dim = input_dim
#         self._learning_rate = learning_rate
#         self._model = self._build_model(num_layers, width)
#         self._optimizer = Adam(lr=self._learning_rate)
#
#     def _build_model(self, num_layers, width):
#         inputs = keras.Input(shape=(self._input_dim,))
#         x = layers.Dense(width, activation='relu')(inputs)
#         for _ in range(num_layers):
#             x = layers.Dense(width, activation='relu')(x)
#         outputs = layers.Dense(1, activation='linear')(x)
#         model = keras.Model(inputs=inputs, outputs=outputs, name='critic_model')
#         return model
#
#     def predict(self, state):
#         return self._model.predict(state, verbose=0)
#
#     def train(self, states, td_targets):
#         td_targets = np.array(td_targets)
#         with tf.GradientTape() as tape:
#             values = self._model(states, training=True)
#             values = tf.squeeze(values)
#             loss = tf.keras.losses.MSE(td_targets, values)
#         # grads = tape.gradient(loss, self._model.trainable_variables)
#         # self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
#         # In ActorModel's train method
#         grads = tape.gradient(loss, self._model.trainable_variables)
#         clipped_grads = [tf.clip_by_norm(g, 1.0) for g in grads]
#         self._optimizer.apply_gradients(zip(clipped_grads, self._model.trainable_variables))

