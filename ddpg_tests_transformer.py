#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# Deep Deterministic Policy Gradient for Portfolio Management
#  
# 1. We have already overfit the model by training on 16 stocks using 3 years of training data 
# Training was done by running stock_traindg.py using DDPG lstm 3 steps window on these 16 stocks
# ['AAPL', 'ATVI', 'CMCSA', 'COST', 'CSX', 'DISH', 'EA', 'EBAY', 'FB', 'GOOGL', 'HAS', 'ILMN', 'INTC', 'MAR', 'REGN', 'SBUX']
#
# 2. We then load a second dataset with different 16 stocks OHLC data
# 3. We test the model performance on the new stocks data (not seen during training) for the training time period
# 4. We then test the model performance on the new stocks data for the testing time period (neither was seen during training)


# In[2]:




import os
os.environ["KERAS_BACKEND"] = "theano"
import numpy as np
from utils.data import read_stock_history, index_to_date, date_to_index, normalize
import matplotlib.pyplot as plt
import matplotlib

import seaborn as sns

# Import Bokeh modules for interactive plotting
import bokeh.io
#import bokeh.mpl
import bokeh.plotting

matplotlib.rcParams['figure.figsize'] = (10, 6)
plt.rc('legend', fontsize=20)
# configure Seaborn settings 
rc = {'lines.linewidth': 2, 
      'axes.labelsize': 18, 
      'axes.titlesize': 18, 
      'axes.facecolor': 'EFEFD5'}
sns.set_context('notebook', rc=rc)
sns.set_style('darkgrid', rc=rc)

# Set up Bokeh for inline viewing
bokeh.io.output_notebook()


# In[4]:




# read the data and choose the target stocks for training a toy example
history, abbreviation = read_stock_history(filepath='utils/datasets/stocks_history_target.h5')
history = history[:, :, :4]
num_training_time = history.shape[1]
num_testing_time = history.shape[1]
window_length = 3


# In[5]:



# dataset for 16 stocks by splitting timestamp
history, abbreviation = read_stock_history(filepath='utils/datasets/stocks_history_target.h5')
history = history[:, :, :4]

# 16 stocks are all involved. We choose first 3 years as training data
num_training_time = 1095
target_stocks = abbreviation
target_history = np.empty(shape=(len(target_stocks), num_training_time, history.shape[2]))

for i, stock in enumerate(target_stocks):
    target_history[i] = history[abbreviation.index(stock), :num_training_time, :]

# and last 2 years as testing data.
testing_stocks = abbreviation
testing_history = np.empty(shape=(len(testing_stocks), history.shape[1] - num_training_time, 
                               history.shape[2]))
for i, stock in enumerate(testing_stocks):
    testing_history[i] = history[abbreviation.index(stock), num_training_time:, :]
print( abbreviation)
print("target_history.shape stocks, steps, OHLC", target_history.shape)
print("testing_history.shape stocks, steps, OHLC", testing_history.shape)


# In[6]:



nb_classes = len(target_stocks) + 1

# visualize stock prices
if False:
    date_list = [index_to_date(i) for i in range(target_history.shape[1])]
    x = range(target_history.shape[1])
    for i in range(len(target_stocks)):
        plt.figure(i)
        plt.plot(x, target_history[i, :, 1])  # open, high, low, close = [0, 1, 2, 3]
        plt.xticks(x[::200], date_list[::200], rotation=30)
        plt.title(target_stocks[i])
        plt.show()


# In[13]:



import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from model.ddpg.actor import ActorNetwork
from model.ddpg.critic import CriticNetwork
from model.ddpg.ddpg import DDPG
from model.ddpg.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise

import numpy as np
import tflearn

from stock_trading import StockActor, StockCritic, obs_normalizer, multi_feature_obs_normalizer, get_model_path, get_result_path,test_model, get_variable_scope, test_model_multiple
    
from model.supervised.lstm import StockLSTM
from model.supervised.cnn import StockCNN
tf.__version__


# In[14]:




# common settings
batch_size = 64
action_bound = 1.
tau = 1e-3

models = []
model_names = []
window_length_lst = [3]
predictor_type_lst = ['lstm']
use_batch_norm = True


# instantiate environment, 16 stocks, with trading cost, window_length 3, start_date sample each time
for window_length in window_length_lst:
    for predictor_type in predictor_type_lst:
        name = 'DDPG_{}_Performance'.format(predictor_type)
        model_names.append(name)
        tf.reset_default_graph()
        sess = tf.Session()
        tflearn.config.init_training_mode()
        action_dim = [nb_classes]
        state_dim = [nb_classes, window_length]
        variable_scope = get_variable_scope(window_length, predictor_type, use_batch_norm)
        with tf.variable_scope(variable_scope):
            actor = StockActor(sess, state_dim, action_dim, action_bound, 1e-4, tau, batch_size, predictor_type, 
                               use_batch_norm)
            critic = StockCritic(sess=sess, state_dim=state_dim, action_dim=action_dim, tau=1e-3,
                                 learning_rate=1e-3, num_actor_vars=actor.get_num_trainable_vars(), 
                                 predictor_type=predictor_type, use_batch_norm=use_batch_norm)
            actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

            model_save_path = get_model_path(window_length, predictor_type, use_batch_norm)
            summary_path = get_result_path(window_length, predictor_type, use_batch_norm)

            ddpg_model = DDPG(None, sess, actor, critic, actor_noise, obs_normalizer=multi_feature_obs_normalizer,
                              config_file='config/stock.json', model_save_path=model_save_path,
                              summary_path=summary_path)
            ddpg_model.initialize(load_weights=True, verbose=True)
            models.append(ddpg_model)


# In[15]:



# create a second dataset for testing that contains different stocks from training stocks shown above
history, abbreviation = read_stock_history(filepath='utils/datasets/stocks_history_target_2.h5')
history = history[:, :, :4]
nb_classes = len(history) + 1
print(history.shape)
print('second stock dataset', abbreviation)
testing_history = history
testing_stocks = abbreviation
target_history = history
target_stocks = abbreviation


# In[16]:


from environment.portfolio import PortfolioEnv, MultiActionPortfolioEnv


# In[17]:


# evaluate the model with dates seen in training but from the second different stocks dataset
env = MultiActionPortfolioEnv(target_history, target_stocks, model_names[:1], steps=1500, 
                              sample_start_date='2012-10-30')

observations_list, info_list, actions_list, df_performance = test_model_multiple(env, models[:1])


# In[18]:



# evaluate the model with unseen dates from the second different stocks dataset, fixed the starting date
env = MultiActionPortfolioEnv(testing_history, testing_stocks, model_names[:1], steps=650, 
                              start_idx=num_training_time, sample_start_date='2016-12-8')

observations_list, info_list, actions_list, df_performance = test_model_multiple(env, models[:1])


# In[19]:MultiActionPortfolioEnv


testing_history


# In[20]:



# evaluate the model with unseen dates from the second different stocks dataset, fixed the starting date
env = MultiActionPortfolioEnv(testing_history, testing_stocks, model_names[:1], steps=650, 
                              start_idx=num_training_time, sample_start_date='2016-12-8')
#2016-12-8
observations_list, info_list, actions_list, df_performance = test_model_multiple(env, models[:1])


# In[21]:


import pandas as pd
actions_list = np.array(actions_list).squeeze(1)
abbreviation.insert(0,'blank')
df_actions = pd.DataFrame(actions_list, columns=abbreviation)
print('stock symbols: ', abbreviation)#[ np.argmax(actions_list[0]) -1])
maxValueIndexObj = df_actions.idxmax(axis=1)
print(' ')
print("Stock selected by the agent for each step :")
print(maxValueIndexObj)


# In[22]:


df_performance = df_performance.reset_index()
maxValueIndexObj.name="stocks"
result = pd.concat([ maxValueIndexObj, df_performance.reindex(maxValueIndexObj.index)], axis=1)

print(result)


# In[23]:


# histogram of the stocks selected by the agent
#maxValueIndexObj.hist()
#maxValueIndexObj.unique()


# In[24]:


# show the stocks selected by the agent at each step in the test period
import matplotlib.pyplot as plt
ax = plt.gca()

df_fox = result.loc[result['stocks'] == 'FOX']
df_ctsh = result.loc[result['stocks'] == 'CTSH']
df_pcln = result.loc[result['stocks'] == 'PCLN']
df_fisv = result.loc[result['stocks'] == 'FISV']
df_fox.index.name = 'idx'
df_ctsh.index.name = 'idx'
df_pcln.index.name = 'idx'
df_fisv.index.name = 'idx'


df= df_fox.reset_index()
df.plot(kind='scatter',x='idx',y='DDPG_window_3_predictor_transformer', color='blue', ax=ax, s=3, label='FOX')
df= df_fisv.reset_index()
df.plot(kind='scatter',x='idx', y='DDPG_window_3_predictor_transformer', color='red', ax=ax, s=3,label='FISV')
df= df_pcln.reset_index()
df.plot(kind='scatter',x='idx', y='DDPG_window_3_predictor_transformer', color='brown', ax=ax, s=3,label='PCLN')
df= df_ctsh.reset_index()
df.plot(kind='scatter',x='idx', y='DDPG_window_3_predictor_transformer', color='green', ax=ax, s=3,label='CTSH')
ax.legend()

#plt.show()


# In[ ]:




