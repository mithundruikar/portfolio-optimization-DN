"""
Use DDPG to train a stock trader based on a window of history price
"""

from __future__ import print_function, division

from model.ddpg.actor import ActorNetwork
from model.ddpg.critic import CriticNetwork
from model.ddpg.ddpg import DDPG
from model.ddpg.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise

from environment.portfolio import PortfolioEnv
from utils.data import read_stock_history, normalize

import numpy as np
import tflearn
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import argparse
import pprint

DEBUG = True


def get_model_path(window_length, predictor_type, use_batch_norm):
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'
    return 'weights/stock/{}/window_{}/{}/checkpoint.ckpt'.format(predictor_type, window_length, batch_norm_str)


def get_result_path(window_length, predictor_type, use_batch_norm):
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'
    return 'results/stock/{}/window_{}/{}/'.format(predictor_type, window_length, batch_norm_str)


def get_variable_scope(window_length, predictor_type, use_batch_norm):
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'
    return '{}_window_{}_{}'.format(predictor_type, window_length, batch_norm_str)

num_layers = 8
num_heads = 3
dropout_rate = 0.1
hidden_dim = 32
batch_size = 64
action_bound = 1.
tau = 1e-4
steps = 730
num_training_time = 1095
default_window_length = 30

def stock_predictor(inputs, predictor_type, use_batch_norm):
    window_length = inputs.get_shape()[2]
    assert predictor_type in ['cnn', 'lstm', 'transformer'], 'type must be either cnn or lstm or transformer'
    if predictor_type == 'cnn':
        net = tflearn.conv_2d(inputs, 32, (1, 3), padding='valid')
        if use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.conv_2d(net, 32, (1, window_length - 2), padding='valid')
        if use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        if DEBUG:
            print('After conv2d:', net.shape)
        net = tflearn.flatten(net)
        if DEBUG:
            print('Output:', net.shape)
    elif predictor_type == 'lstm':
        num_stocks = inputs.get_shape()[1]
        net = tflearn.fully_connected(inputs, num_stocks * window_length, activation='relu')
        hidden_dim = 32
        net = tflearn.reshape(net, new_shape=[-1, window_length, 1])
        if DEBUG:
            print('Reshaped input:', net.shape)
        net = tflearn.lstm(net, hidden_dim) # [batch_size, hidden_dim]
        if DEBUG:
            print('After LSTM:', net.shape)
        net = tflearn.reshape(net, new_shape=[-1, num_stocks, hidden_dim])  # [batch_size, num_stocks, hidden_dim]
        if DEBUG:
            print('After reshape:', net.shape)
        net = tflearn.flatten(net) # [batch_size, num_stocks * hidden_dim]
        if DEBUG:
            print('Output:', net.shape)
    elif predictor_type == 'transformer':
        #inputs : [batch_size, num_assets, window_size, num_features]
        num_stocks = inputs.get_shape()[1]
        hidden_dim = 32
        net = tflearn.fully_connected(inputs, num_stocks * window_length, activation='relu')
        # net: [batch_size, num_assets * window_size]
        net = tflearn.reshape(net, new_shape=[-1, num_stocks, window_length])
        # net: [batch_size, num_assets, window_size]
        if DEBUG:
            print('Reshaped input:', net.shape)
        #net = tflearn.lstm(net, hidden_dim)
        _, net = build_transformer_encoder(net, num_layers, num_heads, hidden_dim, dropout_rate) # [batch_size, hidden_dim]
        #net = tflearn.reshape(net, new_shape=[-1, num_stocks, hidden_dim])  # [batch_size, num_stocks, hidden_dim]
        if DEBUG:
            print('After reshape:', net.shape)
        #net = tflearn.flatten(net)  # [batch_size, num_stocks * hidden_dim]
        net = tflearn.fully_connected(net, num_stocks * hidden_dim, activation='relu')
        if DEBUG:
            print('Output:', net.shape)
    else:
        raise NotImplementedError

    return net

def build_transformer_encoder(inputs, num_layers, num_heads, ff_dim, dropout_rate=0.1):
    x = inputs
    for _ in range(num_layers):
        x = transformer_encoder_block(x, num_heads, ff_dim, dropout_rate)

    #ff_output = tflearn.fully_connected(x, ff_dim, activation='relu')  # [?, 32]
    #encoder_output = tflearn.layers.normalization.batch_normalization(ff_output)  # [?, 32]
    #return inputs, tflearn.layers.core.dropout(encoder_output, dropout_rate)  # [?, 32]

    return inputs, x

def transformer_encoder_block(inputs, num_heads, ff_dim, dropout_rate=0.1):
    # Multi-head self-attention layer
    attention_output = multi_head_self_attention(inputs, inputs, inputs, num_heads) # [?, 3]

    # Add and normalize
    attention_output = tflearn.layers.normalization.batch_normalization(attention_output)
    # Reshape attention_output to have the same shape as inputs
    attention_output = tf.expand_dims(attention_output, axis=1)
    attention_output = tf.tile(attention_output, [1, inputs.shape[1], 1]) # [?, num_assets, 3]
    attention_output = tf.add(inputs, attention_output)

    #attention_output = tflearn.layers.normalization.batch_normalization(attention_output)
    #return tflearn.layers.core.dropout(attention_output, dropout_rate)

    # Feed-forward neural network
    ff_output = tflearn.fully_connected(attention_output, ff_dim, activation='relu') # [?, 32]
    #encoder_output = tflearn.layers.normalization.batch_normalization(ff_output)  # [?, 32]
    #return  tflearn.layers.core.dropout(encoder_output, dropout_rate) # [?, 32]

    ff_output = tflearn.fully_connected(ff_output, inputs.shape[-1]) # [?, 3]

    # Add and normalize
    encoder_output = tflearn.layers.normalization.batch_normalization(ff_output) # [?, 3]
    encoder_output = tf.expand_dims(encoder_output, axis=1)
    encoder_output = tf.tile(encoder_output, [1, inputs.shape[1], 1])  # [?, num_assets, 3]
    encoder_output = tf.add(attention_output, encoder_output)
    encoder_output = tflearn.layers.core.dropout(encoder_output, dropout_rate)

    return encoder_output # [?, num_assets, 3]

def multi_head_self_attention(q, k, v, num_heads):
    d_model = q.shape[-1]
    assert d_model % num_heads == 0

    depth = d_model // num_heads

    # Linear layers for Query, Key, and Value
    query = tflearn.fully_connected(q, d_model)
    key = tflearn.fully_connected(k, d_model)
    value = tflearn.fully_connected(v, d_model)

    # Split into multiple heads
    query = tf.reshape(query, (-1, num_heads, depth))
    key = tf.reshape(key, (-1, num_heads, depth))
    value = tf.reshape(value, (-1, num_heads, depth))

    # Scaled dot-product attention
    attention_scores = tf.matmul(query, key, transpose_b=True)
    attention_scores = attention_scores / tf.sqrt(tf.cast(depth, tf.float32))
    attention_scores = tf.nn.softmax(attention_scores, axis=-1)
    output = tf.matmul(attention_scores, value)

    # Reshape and concatenate heads
    output = tf.reshape(output, (-1, d_model))
    return output


class StockActor(ActorNetwork):
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size,
                 predictor_type, use_batch_norm):
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        ActorNetwork.__init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size)

    def create_actor_network(self):
        """
        self.s_dim: a list specifies shape
        """
        nb_classes, window_length = self.s_dim
        assert nb_classes == self.a_dim[0]
        assert window_length > 2, 'This architecture only support window length larger than 2.'
        inputs = tflearn.input_data(shape=[None] + self.s_dim + [4], name='input')

        net = stock_predictor(inputs, self.predictor_type, self.use_batch_norm)


        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, self.a_dim[0], activation='softmax', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })


class StockCritic(CriticNetwork):
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars,
                 predictor_type, use_batch_norm):
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        CriticNetwork.__init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None] + self.s_dim + [4])
        action = tflearn.input_data(shape=[None] + self.a_dim)

        net = stock_predictor(inputs, self.predictor_type, self.use_batch_norm)
        #net = tf.Print(net, [tf.shape(net)], "net after stock prediction")
        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 64)
        #action_scaled = tf.repeat(action, repeats=1, axis=0)
        t2 = tflearn.fully_connected(action, 64)
        #t1 = tf.Print(t1, [tf.shape(t1)], "t1 value before addition")
        #t2 = tf.Print(t2, [tf.shape(t2)], "t2 value before addition")
        net = tf.add(t1, t2)
        if self.use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })


def obs_normalizer(observation):
    """ Preprocess observation obtained by environment

    Args:
        observation: (nb_classes, window_length, num_features) or with info

    Returns: normalized

    """
    if isinstance(observation, tuple):
        observation = observation[0]
    # directly use close/open ratio as feature
    observation = observation[:, :, 3:4] / observation[:, :, 0:1]
    observation = normalize(observation)
    return observation

def multi_feature_obs_normalizer(observation):
    """ Preprocess observation obtained by environment

    Args:
        observation: (nb_classes, window_length, num_features) or with info

    Returns: normalized

    """
    if isinstance(observation, tuple):
        observation = observation[0]

    observation = normalize(observation)
    return observation


def test_model(env, model):
    observation, info = env.reset()
    done = False
    while not done:
        action = model.predict_single(observation)
        observation, _, done, _ = env.step(action)
    env.render()


def test_model_multiple(env, models):
    observations_list = []
    actions_list = []
    info_list = []
    observation, info = env.reset()
    done = False
    while not done:
        actions = []
        for model in models:
            actions.append(model.predict_single(observation))
        actions = np.array(actions)
        observation, _, done, info = env.step(actions)
        observations_list.append(observation)
        actions_list.append(actions)
        info_list.append(info)
    df_performance = env.render()
    return observations_list, info_list, actions_list, df_performance


import psutil
import matplotlib.pyplot as plt
import time

# Lists to store resource usage data
cpu_usage = []
memory_usage = []


# Function to monitor and record CPU and memory usage
def monitor_resource_usage(interval_seconds, duration_seconds):
    start_time = time.time()
    end_time = start_time + duration_seconds

    while time.time() < end_time:
        cpu_percent = psutil.cpu_percent(interval=interval_seconds)
        memory_percent = psutil.virtual_memory().percent

        cpu_usage.append(cpu_percent)
        memory_usage.append(memory_percent)
        time.sleep(interval_seconds)


# Example usage
monitor_resource_usage(interval_seconds=1, duration_seconds=60)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Provide arguments for training different DDPG models')

    parser.add_argument('--debug', '-d', help='print debug statement', default=True)
    parser.add_argument('--predictor_type', '-p', help='cnn or lstm predictor', default='lstm')
    parser.add_argument('--window_length', '-w', help='observation window length', default=default_window_length)
    parser.add_argument('--batch_norm', '-b', help='whether to use batch normalization', default='True')

    args = vars(parser.parse_args())

    pprint.pprint(args)

    if args['debug'] == 'True':
        DEBUG = True
    else:
        DEBUG = False

    history, abbreviation = read_stock_history(filepath='utils/datasets/stocks_history_target.h5')
    history = history[:, :, :4]
    target_stocks = abbreviation
    window_length = int(args['window_length'])
    nb_classes = len(target_stocks) + 1

    # get target history
    target_history = np.empty(shape=(len(target_stocks), num_training_time, history.shape[2]))
    for i, stock in enumerate(target_stocks):
        target_history[i] = history[abbreviation.index(stock), :num_training_time, :]

    # setup environment
    env = PortfolioEnv(target_history, target_stocks, steps=steps, window_length=window_length)

    action_dim = [nb_classes]
    state_dim = [nb_classes, window_length]

    assert args['predictor_type'] in ['cnn', 'lstm', 'transformer'], 'Predictor must be either cnn or lstm'
    predictor_type = args['predictor_type']
    if args['batch_norm'] == 'True':
        use_batch_norm = True
    elif args['batch_norm'] == 'False':
        use_batch_norm = False
    else:
        raise ValueError('Unknown batch norm argument')
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
    model_save_path = get_model_path(window_length, predictor_type, use_batch_norm)
    summary_path = get_result_path(window_length, predictor_type, use_batch_norm)

    variable_scope = get_variable_scope(window_length, predictor_type, use_batch_norm)

    with tf.variable_scope(variable_scope):
        sess = tf.Session()
        actor = StockActor(sess, state_dim, action_dim, action_bound, 1e-4, tau, batch_size,
                           predictor_type, use_batch_norm)
        critic = StockCritic(sess=sess, state_dim=state_dim, action_dim=action_dim, tau=tau,
                             learning_rate=tau, num_actor_vars=actor.get_num_trainable_vars(),
                             predictor_type=predictor_type, use_batch_norm=use_batch_norm)
        ddpg_model = DDPG(env, sess, actor, critic, actor_noise, obs_normalizer=multi_feature_obs_normalizer, #obs_normalizer=obs_normalizer,
                          config_file='config/stock.json', model_save_path=model_save_path,
                          summary_path=summary_path)
        ddpg_model.initialize(load_weights=False)
        print('calling DDPG train')
        ddpg_model.train()

    env.render()


    plt.subplot(2, 1, 2)
    plt.plot(cpu_usage, label='CPU Usage (%)', color='blue')
    plt.plot(memory_usage, label='Memory Usage (%)', color='green')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Usage (%)')
    plt.legend()
    plt.title('CPU and Memory Usage During Training')
    plt.show()
