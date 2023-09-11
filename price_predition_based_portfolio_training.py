import datetime
from pandas_datareader import data as pdr
import yfinance as yfin
yfin.pdr_override()

import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
from model.lstm import LSTM_model
from model.transformer import TransAm
import matplotlib.pyplot as plt
import time
import math

# getting datetimeindex

# except ABBV, FB, TSLA: listed after 2009-12-31
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'BRK-A', 'NVDA', 'V', 'JPM', 'UNH',
           'JNJ', 'BAC', 'WMT', 'PG', 'HD', 'MA', 'XOM', 'PFE', 'DIS', 'CVX',
           'KO', 'AVGO', 'PEP', 'CSCO', 'WFC', 'COST', 'LLY', 'ADBE']
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
#tickers = ['AAPL', 'AMZN']
start_time = '2009-12-31'  # 2009-12-31
end_time = '2021-12-31'  # 2021-12-31
all_split = 2517  # start 31/12/2009, val 02/01/2020, end 31/12/2021
# selecting gpu
device = torch.device('cuda')
risk_free_ret = 0.015
time_step = 30
num_epochs = 1000
batch_size = 100
val_loss_buffer = 0.001  # You can adjust this value based on your needs
max_consecutive_increases = 3
# scaler for normalizing dataset
scaler = MinMaxScaler()
n_assets = len(tickers)
eq_weights = np.ones(n_assets) / n_assets

def main():
    loadData()
    build_model()
    #fit_transform_1()
    start_time = time.time()
    fit_transform()
    print(f"training time: {( time.time() - start_time) } seconds")
    #fit_transform_no_batch()
    start_time = time.time()
    predict()
    plot_model_perf()
    print(f"prediction time: {(time.time() - start_time)} seconds")
    postTrainCleanup()
    buildPortfolios()
    #plotPortfolios()
    plotPortfolios_pred()
    plot_sharpe_ration()


def loadData(loadFromSource = False):
# dataframe for Adj Close price
    global act_adj_close_df
    if loadFromSource:
        act_adj_close_df = pd.DataFrame(index=[0], columns=tickers)
        for ticker in tickers:
            stock_price_df = pdr.get_data_yahoo(ticker, start_time, end_time)
            stock_price_df.index = pd.to_datetime(stock_price_df.index)
            act_adj_close_df[ticker] = stock_price_df['Adj Close']
    else:
        act_adj_close_df = pd.read_csv('data/act-adj-close-df.csv', index_col=0)
        act_adj_close_df.index = pd.to_datetime(act_adj_close_df.index)
        act_adj_close_df = act_adj_close_df[tickers]

    global model_loss_df, pred_adj_close_df, model_perf_df, loss_df, val_loss_df
    # dataframes for model loss, predicted Adj Close price and model performance
    model_loss_df = pd.DataFrame(index=[epoch for epoch in range(0, num_epochs*11, num_epochs)], columns=tickers)
    val_loss_df = pd.DataFrame(index=range(0, int((act_adj_close_df.shape[0] - all_split - time_step - 2) / batch_size + 1)), columns=tickers)

    # pred_adj_close_df = pd.DataFrame(index=df.index[time_step + 1:], columns=tickers)
    pred_adj_close_df = pd.DataFrame(index=act_adj_close_df.index[time_step + 1 + all_split:], columns=tickers)
    model_perf_df = pd.DataFrame(index=['MAE', 'MSE', 'RMSE', 'MAPE', 'MPE'], columns=tickers)
    #loss_df = pd.DataFrame(index=range(act_adj_close_df.shape[0]-all_split-time_step - 1), columns=tickers)


# function for building model
def build_model(input_size=1, hidden_size=1, num_layers=1, num_classes=1, learning_rate=0.0001):
    # learning rate controls how much to change model in response to estm error 
    # each time model weights are updated

    global model, optimizer, loss_function
    #model = LSTM_model(input_size, hidden_size, num_layers, num_classes, device).to(device)
    model = TransAm(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # adam optimizer
    #optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # adam optimizer
    # algorithms/methods to change attributes of neural network such as weights and learning rate to reduce losses

    loss_function = torch.nn.MSELoss() # mean-squared error of regression
    # loss function measures how bad model performs: high loss -> low accuracy

    # loading model state
    # model = LSTM_model(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    # model.eval()

    return model, optimizer, loss_function

# function for creating X and y
def create_xy(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:i + time_step])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)


# functions for model performance evaluation
def MAE(y_true, y_hat):
    return np.mean(np.abs(y_true - y_hat))

def MSE(y_true, y_hat):
    return np.mean(np.square(y_true - y_hat))

def RMSE(y_true, y_hat):
    return np.sqrt(MSE(y_true, y_hat))

def MAPE(y_true, y_hat):
    return np.mean(np.abs((y_true - y_hat) / y_true)) * 100

def MPE(y_true, y_hat):
    return np.mean((y_true - y_hat) / y_true) * 100


# function for fitting model
def fit_model_batch(X_train=None, y_train=None):
    outputs = model.forward(X_train.to(device))
    optimizer.zero_grad()  # calculating gradient, manually setting to 0
    loss = loss_function(outputs, y_train.to(device))  # obtaining loss
    loss.backward()  # calculating loss of loss function
    optimizer.step()  # improving from loss, i.e. backprop
    return loss



# function for fitting model
def fit_model(ticker, X_train, y_train):
    loss_list = []
    for epoch in range(num_epochs + 1):
        outputs = model.forward(X_train.to(device))
        #outputs = model.forward(X_train.to(device).view([10, X_train.shape[0], 1])) # forward pass
        optimizer.zero_grad() # calculating gradient, manually setting to 0
        #loss = loss_function(outputs, y_train.to(device).view([1, y_train.shape[0], 1])) # obtaining loss
        loss = loss_function(outputs, y_train.to(device))  # obtaining loss
        loss.backward() # calculating loss of loss function
        optimizer.step() # improving from loss, i.e. backprop


        if epoch % (num_epochs/10) == 0:
            loss_list.append(loss.item())
            print('loss iter ', epoch, loss.item())



    print('completed epochs of the train run')
    model_loss_df[ticker] = loss_list

    # saving model state
    torch.save(model.state_dict(), f'data/model-states/{ticker}-model-state.pth')
    
    return model




# fitting models and predicting responses
def fit_transform(val_loss_check = False):
    for ticker in tickers:
    #for ticker in [tickers[0]]:
        # normalizing dataset
        norm_act_adj_close = scaler.fit_transform(np.array(act_adj_close_df[ticker]).reshape(-1, 1))

        # if ticker == 'ABBV':
        #     train_set, val_set = norm_act_adj_close[:abbv_split], norm_act_adj_close[abbv_split:]
        # elif ticker == 'FB':
        #     train_set, val_set = norm_act_adj_close[:fb_split], norm_act_adj_close[fb_split:]
        # elif ticker == 'TSLA':
        #     train_set, val_set = norm_act_adj_close[:tsla_split], norm_act_adj_close[tsla_split:]
        # else:
        val_batch_count = 1
        train_set, val_set = norm_act_adj_close[:all_split], norm_act_adj_close[all_split:all_split+ (batch_size*val_batch_count)]

        X_train, y_train = create_xy(train_set, time_step)

        # converting datasets to tensors
        X_train_tensors = Variable(torch.Tensor(X_train))
        y_train_tensors = Variable(torch.Tensor(y_train))

        X_val, y_val = create_xy(val_set, time_step)

        # converting datasets to tensors
        X_val_tensors = Variable(torch.Tensor(X_val))
        y_val_tensors = Variable(torch.Tensor(y_val))

        # fitting model
        print('----------')
        print(f'{ticker} fitting...')
        print(f'{len(tickers) - (tickers.index(ticker) + 1)} left')

        loss_list = []
        previous_val_loss = float('inf')
        consecutive_increases = 0
        for epoch in range(num_epochs + 1):
            for batch, i in enumerate(range(0, len(X_train_tensors) - 1, batch_size)):
                #print('training batch sequence ', i)
                X_train_tensors_batched, y_train_tensors_batched = getBatch(X_train_tensors, y_train_tensors, i, batch_size)
                loss = fit_model_batch(X_train=X_train_tensors_batched, y_train=y_train_tensors_batched)

            if val_loss_check:
                val_loss = 0
                for batch, i in enumerate(range(0, val_batch_count)):
                    #print('training batch sequence ', i)
                    X_val_tensors_batched, y_val_tensors_batched = getBatch(X_val_tensors, y_val_tensors, i, batch_size)
                    pred_out = model(X_val_tensors_batched.to(device))
                    val_loss += loss_function(pred_out, y_val_tensors_batched.to(device)).item()

                average_validation_loss = val_loss / val_batch_count
                # Check for overfitting with a buffer
                if average_validation_loss > (previous_val_loss + val_loss_buffer):
                    consecutive_increases += 1
                else:
                    consecutive_increases = 0

                previous_val_loss = average_validation_loss

                if consecutive_increases >= max_consecutive_increases:
                    print("Early Stopping Triggered!")
                    break

            if epoch % (num_epochs / 10) == 0:
                loss_list.append(loss.item())
                print('loss iter ', epoch, loss.item())

        print('completed epochs of the train run')
        model_loss_df[ticker] = extend_list_with_nan(loss_list, model_loss_df.shape[0])

            # saving model state
            #torch.save(model.state_dict(), f'data/model-states/{ticker}-model-state.pth')

    print('Done training!')
    plot_losses()


def extend_list_with_nan(input_list, target_size):
    if target_size <= len(input_list):
        return input_list

    nan_list = [math.nan] * (target_size - len(input_list))
    extended_list = input_list + nan_list

    return extended_list

# fitting models and predicting responses
def predict():
    for ticker in tickers:
    #for ticker in [tickers[0]]:
        # normalizing dataset
        norm_act_adj_close = scaler.fit_transform(np.array(act_adj_close_df[ticker]).reshape(-1, 1))

        X_val, y_val = create_xy(norm_act_adj_close[all_split:], time_step)

        # converting datasets to tensors
        X_val_tensors = Variable(torch.Tensor(X_val))
        y_val_tensors = Variable(torch.Tensor(y_val))
        y_true = scaler.inverse_transform(y_val) # inverse transformation

        y_hat = torch.randn(0, 1).to(device)
        val_loss = []
        for batch, i in enumerate(range(0, len(X_val_tensors) - 1, batch_size)):
            # predicting response
            #print('predicting batch sequence ', i)
            X_val_tensors_batched, y_val_tensors_batched = getBatch(X_val_tensors, y_val_tensors, i, batch_size)
            pred_out = model(X_val_tensors_batched.to(device))
            loss = loss_function(pred_out, y_val_tensors_batched.to(device))  # obtaining loss
            val_loss.append(loss.item())
            y_hat = torch.cat((y_hat, pred_out), dim=0)  # forward pass

        val_loss_df[ticker] = cumulative_avg_np(np.array(val_loss))
        y_hat = y_hat.data.detach().cpu().numpy() # numpy conversion
        y_hat = scaler.inverse_transform(y_hat) # inverse transformation
        print('Done Predicting!')

        pred_adj_close_df[ticker] = y_hat

        model_perf_df[ticker] = [MAE(y_true, y_hat), MSE(y_true, y_hat), RMSE(y_true, y_hat),
                                 MAPE(y_true, y_hat), MPE(y_true, y_hat)]
    plot_val_losses()

def cumulative_avg_np(arr):
    cumulative_sum = np.cumsum(arr)
    cumulative_count = np.arange(1, len(arr) + 1)
    cumulative_avg = cumulative_sum / cumulative_count
    return cumulative_avg

# fitting models and predicting responses
def fit_transform_no_batch():
    for ticker in tickers:
    #for ticker in [tickers[0]]:
        # normalizing dataset
        norm_act_adj_close = scaler.fit_transform(np.array(act_adj_close_df[ticker]).reshape(-1, 1))
        # if ticker == 'ABBV':
        #     train_set, val_set = norm_act_adj_close[:abbv_split], norm_act_adj_close[abbv_split:]
        # elif ticker == 'FB':
        #     train_set, val_set = norm_act_adj_close[:fb_split], norm_act_adj_close[fb_split:]
        # elif ticker == 'TSLA':
        #     train_set, val_set = norm_act_adj_close[:tsla_split], norm_act_adj_close[tsla_split:]
        # else:
        train_set, val_set = norm_act_adj_close[:all_split], norm_act_adj_close[all_split:]

        X_train, y_train = create_xy(train_set, time_step)
        X_val, y_val = create_xy(norm_act_adj_close, time_step)

        # converting datasets to tensors
        X_train_tensors = Variable(torch.Tensor(X_train))
        y_train_tensors = Variable(torch.Tensor(y_train))

        X_val_tensors = Variable(torch.Tensor(X_val))
        y_true = scaler.inverse_transform(y_val) # inverse transformation

        # fitting model
        print('----------')
        print(f'{ticker} fitting...')
        print(f'{len(tickers) - (tickers.index(ticker) + 1)} left')
        model = fit_model(ticker=ticker, X_train=X_train_tensors, y_train=y_train_tensors)

        # predicting response
        y_hat = model(X_val_tensors.to(device)) # forward pass
        y_hat = y_hat.data.detach().cpu().numpy() # numpy conversion
        y_hat = scaler.inverse_transform(y_hat) # inverse transformation
        print('Done!')

        pred_adj_close_df[ticker] = y_hat

        model_perf_df[ticker] = [MAE(y_true, y_hat), MSE(y_true, y_hat), RMSE(y_true, y_hat),
                                 MAPE(y_true, y_hat), MPE(y_true, y_hat)]
    plot_losses()


def getBatch(inputTensor, labelsTensor, startIndex, batchSize):
    seq_len = min(batchSize, len(inputTensor) - startIndex)
    return inputTensor[startIndex:startIndex+seq_len], labelsTensor[startIndex:startIndex+seq_len]

def saveProgress():
    model_loss_df.to_csv('data/model-loss-df.csv')
    pred_adj_close_df.to_csv('data/pred-adj-close-df.csv')
    model_perf_df.to_csv('data/model-perf-df.csv')

def postTrainCleanup():
    # daily returns of actual Adj Close price (validation phase)
    act_adj_close_val_df = act_adj_close_df.iloc[time_step + 1 + all_split:, :]
    global act_daily_ret_df
    act_daily_ret_df = act_adj_close_val_df.pct_change()
    act_daily_ret_df = act_daily_ret_df.iloc[1:, :]
    act_daily_ret_df.head()


    # daily returns of predicted Adj Close price (validation phase)
    pred_adj_close_val_df = pred_adj_close_df.iloc[:, :]
    global pred_daily_ret_df
    pred_daily_ret_df = pred_adj_close_val_df.pct_change()
    pred_daily_ret_df = pred_daily_ret_df.iloc[1:, :]
    pred_daily_ret_df.head()


    # dataframe for daily portfolio returns
    global act_daily_port_ret_df
    act_daily_port_ret_df = act_daily_ret_df.copy()
    act_daily_port_ret_df.drop(columns=tickers, inplace=True)

    # dataframe for daily portfolio returns
    global pred_daily_port_ret_df
    pred_daily_port_ret_df = pred_daily_ret_df.copy()
    pred_daily_port_ret_df.drop(columns=tickers, inplace=True)


# function for adding different portfolios
def add_portfolio(portfolio, weights):
    act_daily_port_ret_df[f'{portfolio} Return'] = act_daily_ret_df.dot(weights)
    if portfolio == 'Pred':
        # upper bound of security transaction tax: 0.0003
        act_daily_port_ret_df[f'{portfolio} Return'] = act_daily_port_ret_df[f'{portfolio} Return'] - 0.0003
    act_daily_port_ret_df[f'{portfolio} Cum Prod Return'] = (1 + act_daily_port_ret_df[f'{portfolio} Return']).cumprod()

    act_daily_port_ret_df[f'{portfolio} Rolling_Mean'] = act_daily_port_ret_df[f'{portfolio} Return'].rolling(time_step).mean()
    act_daily_port_ret_df[f'{portfolio} Rolling_Std'] = act_daily_port_ret_df[f'{portfolio} Return'].rolling(time_step).std()

    # Calculate the running Sharpe ratio
    act_daily_port_ret_df[f'{portfolio} Sharpe_Ratio'] = act_daily_port_ret_df[f'{portfolio} Rolling_Mean'] / act_daily_port_ret_df[f'{portfolio} Rolling_Std']

    exp_ret = act_daily_port_ret_df[f'{portfolio} Return'].mean()
    std = act_daily_port_ret_df[f'{portfolio} Return'].std()
    sharpe_ratio = (exp_ret - risk_free_ret) / std

    print(portfolio)
    print('Weights:')
    print(weights)
    print(f'Expected Return: {exp_ret:.6f}')
    print(f'Standard Dev   : {std:.6f}')
    print(f'Sharpe Ratio   : {sharpe_ratio:.6f}')


# function for adding different portfolios
def add_portfolio_pred(portfolio, weights):
    pred_daily_port_ret_df[f'{portfolio} Return'] = pred_daily_ret_df.dot(weights)
    if portfolio == 'Pred':
        # upper bound of security transaction tax: 0.0003
        pred_daily_port_ret_df[f'{portfolio} Return'] = pred_daily_port_ret_df[f'{portfolio} Return'] - 0.0003
    pred_daily_port_ret_df[f'{portfolio} Cum Prod Return'] = (1 + pred_daily_port_ret_df[f'{portfolio} Return']).cumprod()

    pred_daily_port_ret_df[f'{portfolio} Rolling_Mean'] = pred_daily_port_ret_df[f'{portfolio} Return'].rolling(time_step).mean()
    pred_daily_port_ret_df[f'{portfolio} Rolling_Std'] = pred_daily_port_ret_df[f'{portfolio} Return'].rolling(time_step).std()

    # Calculate the running Sharpe ratio
    pred_daily_port_ret_df[f'{portfolio} Sharpe_Ratio'] = pred_daily_port_ret_df[f'{portfolio} Rolling_Mean'] / pred_daily_port_ret_df[f'{portfolio} Rolling_Std']

    exp_ret = pred_daily_port_ret_df[f'{portfolio} Return'].mean()
    std = pred_daily_port_ret_df[f'{portfolio} Return'].std()
    sharpe_ratio = (exp_ret - risk_free_ret) / std

    print(portfolio)
    print('Weights:')
    print(weights)
    print(f'Expected Return: {exp_ret:.6f}')
    print(f'Standard Dev   : {std:.6f}')
    print(f'Sharpe Ratio   : {sharpe_ratio:.6f}')

def buildPortfolios():
    eqWeightPortfolio()
    #capDataPortfolio()
    createPreditionPortfolio()
    global act_daily_port_ret_df
    act_daily_port_ret_df = act_daily_port_ret_df[time_step:]
    global pred_daily_port_ret_df
    pred_daily_port_ret_df = pred_daily_port_ret_df[time_step:]

def eqWeightPortfolio():
    #add_portfolio('Eq', eq_weights)
    add_portfolio_pred('Eq', eq_weights)


def capDataPortfolio():
    # #### Market capitalization weighted portfolio
    cap_data = pdr.get_quote_yahoo(tickers)['marketCap']
    cap_df = pd.DataFrame(cap_data)
    cap_df['Weight'] = cap_df / cap_df.sum()
    cap_weights = np.array(cap_df['Weight'])
    add_portfolio('Cap', cap_weights)


# function for getting inverse of expected return using predicted Adj Close price
# used to get optimum weights that maximize this expected return
def exp_ret_inv(weights):
    pred_daily_port_ret = pred_daily_ret_df.dot(weights)
    exp_ret = pred_daily_port_ret.mean()
    return 1 / exp_ret

def createPreditionPortfolio():
    # expected return maximization using predicted Adj Close price
    weights0 = eq_weights # initial weights
    # bounds: weight should be between 0.0 and 1.0
    bnds = tuple((0.0, 1.0) for i in range(n_assets))
    # constraints: weights should add up to 1.0
    cons = ({'type': 'eq', 'fun': lambda W: np.sum(W) - 1.0})

    res = minimize(exp_ret_inv, weights0, method='SLSQP', bounds=bnds, constraints=cons)
    pred_weights = res.x

    #add_portfolio('Pred', pred_weights)
    add_portfolio_pred('Pred', pred_weights)

def plot_losses():
    for ticker in tickers:
        plt.plot(model_loss_df.index.values, model_loss_df[ticker].values,
                 label=ticker+' training loss')
    plt.xlabel('Date')
    plt.ylabel('training loss')
    plt.legend()
    plt.show()

def plot_val_losses():
    for ticker in tickers:
        plt.plot(val_loss_df.index.values, val_loss_df[ticker].values,
                 label=ticker+' Validation loss')
    plt.xlabel('Date')
    plt.ylabel('Val loss')
    plt.legend()
    plt.show()

def plot_model_perf():
    for ticker in tickers:
        plt.plot(model_perf_df.index.values, model_perf_df[ticker].values,
                 label=ticker+' Performance Indicators')
    plt.xlabel('Performance Indicators')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

def plotPortfolios():
    plt.plot(act_daily_port_ret_df.index.values, act_daily_port_ret_df['Eq Cum Prod Return'].values,
             label='Equally Weighted Portfolio', color='fuchsia')
    # plt.plot(act_daily_port_ret_df.index, act_daily_port_ret_df['Cap Cum Prod Return'],
    #         label='Cap Weighted Portfolio', color='blue')
    plt.plot(act_daily_port_ret_df.index.values, act_daily_port_ret_df['Pred Cum Prod Return'].values,
             label='Transformer Predictions Portfolio', color='red')
    plt.xlabel('Date')
    plt.ylabel('Cummulative Product Return')
    plt.title('Cummulative Product Returns')
    #plt.ylim(0.9, 2.5)
    plt.legend()
    plt.show()

def plotPortfolios_pred():
    plt.plot(pred_daily_port_ret_df.index.values, pred_daily_port_ret_df['Eq Cum Prod Return'].values,
             label='Equally Weighted Portfolio', color='fuchsia')
    # plt.plot(act_daily_port_ret_df.index, act_daily_port_ret_df['Cap Cum Prod Return'],
    #         label='Cap Weighted Portfolio', color='blue')
    plt.plot(pred_daily_port_ret_df.index.values, pred_daily_port_ret_df['Pred Cum Prod Return'].values,
             label='Transformer Predictions Portfolio', color='red')
    plt.xlabel('Date')
    plt.ylabel('Cummulative Product Return')
    plt.title('Cummulative Product Returns')
    #plt.ylim(0.9, 2.5)
    plt.legend()
    plt.show()

def plot_sharpe_ration():
    plt.plot(pred_daily_port_ret_df.index.values, pred_daily_port_ret_df['Eq Sharpe_Ratio'].values,
             label='Equally Weighted Sharpe_Ratio', color='fuchsia')
    # plt.plot(act_daily_port_ret_df.index, act_daily_port_ret_df['Cap Cum Prod Return'],
    #         label='Cap Weighted Portfolio', color='blue')
    plt.plot(pred_daily_port_ret_df.index.values, pred_daily_port_ret_df['Pred Sharpe_Ratio'].values,
             label='Transformer Predictions Sharpe_Ratio', color='red')
    plt.xlabel('Date')
    plt.ylabel('Equally Weighted Sharpe_Ratio')
    plt.title('Transformer Predictions Sharpe_Ratio')
    #plt.ylim(0.9, 2.5)
    plt.legend()
    plt.show()


main()