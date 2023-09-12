import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Define the Transformer model for portfolio optimization
def build_portfolio_transformer(num_assets, window_size, num_features_of_stock, num_heads=2, ff_dim=32, num_layers=2):
    input_layer = keras.layers.Input(shape=(window_size, num_features_of_stock))

    # Multi-Head Self-Attention Layer
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=num_features_of_stock)(input_layer, input_layer)
    x = layers.Dropout(0.1)(x)
    residual = layers.Add()([input_layer, x])

    for _ in range(num_layers):
        # Feed Forward Layer
        x = layers.TimeDistributed(layers.Dense(ff_dim, activation="relu"))(residual)
        x = layers.TimeDistributed(layers.Dense(num_features_of_stock))(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Add()([residual, x])

    # Global Average Pooling to get portfolio weights
    avg_pool = layers.GlobalAveragePooling1D()(x)
    output_layer = layers.Dense(num_assets, activation="softmax")(avg_pool)  # Softmax for portfolio weights

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model


# Example usage
num_assets = 10  # Number of assets in the portfolio
window_size = 20  # Historical window size
num_features_of_stock = 5  # Number of features for each stock

model = build_portfolio_transformer(num_assets, window_size, num_features_of_stock)
model.summary()
model
