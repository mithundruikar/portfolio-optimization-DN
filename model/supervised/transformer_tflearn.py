import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tflearn
import numpy as np

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

def transformer_encoder_block(inputs, num_heads, ff_dim, dropout_rate=0.1):
    # Multi-head self-attention layer
    attention_output = multi_head_self_attention(inputs, inputs, inputs, num_heads)

    # Add and normalize
    attention_output = tflearn.layers.normalization.batch_normalization(attention_output)
    # Reshape attention_output to have the same shape as inputs
    attention_output = tf.reshape(attention_output, (-1, 1, 3))
    attention_output = tf.add(inputs, attention_output)

    # Feed-forward neural network
    ff_output = tflearn.fully_connected(attention_output, ff_dim, activation='relu')
    ff_output = tflearn.fully_connected(ff_output, inputs.shape[-1])

    # Add and normalize
    encoder_output = tflearn.layers.normalization.batch_normalization(ff_output)
    encoder_output = tf.add(attention_output, encoder_output)
    encoder_output = tflearn.layers.core.dropout(encoder_output, dropout_rate)

    return encoder_output

def build_transformer_encoder(input_shape, num_layers, num_heads, ff_dim, dropout_rate=0.1):
    inputs = tf.placeholder(tf.float32, shape=(None, *input_shape))
    build_transformer(inputs, num_layers, num_heads, ff_dim, dropout_rate)

def build_transformer(inputs, num_layers, num_heads, ff_dim, dropout_rate=0.1):
    x = inputs
    for _ in range(num_layers):
        x = transformer_encoder_block(x, num_heads, ff_dim, dropout_rate)

    return inputs, x


# Assuming inputs and attention_output are your tensors
inputs = tf.constant([[1.0] * 51], shape=(64, 17, 3), dtype=tf.float32)  # Example inputs
attention_output = tf.constant([1.0] * 3, shape=(64, 3), dtype=tf.float32)  # Example attention_output

# Reshape attention_output to have the same shape as inputs
attention_output = tf.expand_dims(attention_output, axis=1)
attention_output = tf.tile(attention_output, [1, 17, 1])

#attention_output = tf.reshape(attention_output, (inputs.shape[0], inputs.shape[1], 3))

# Now, you can perform element-wise addition
output = tf.add(inputs, attention_output)

with tf.Session() as sess:
    print(inputs.eval())
    print(attention_output.eval())
    print(output.eval())

# Check the shape of the output
print(output.shape)  # It should be (1, 17, 3)


# Example usage
#input_shape = (512, 64)  # Sequence length and embedding dimension
input_shape = (17, 3, 1)
num_layers = 16
num_heads = 3
ff_dim = 2048
dropout_rate = 0.1
hidden_dim = 32

#inputs = tf.placeholder(tf.float32, shape=(None, *input_shape))
inputs = tflearn.input_data(shape=[None, 17, 3, 1])
net = tflearn.reshape(inputs, new_shape=[-1, inputs.get_shape()[1], inputs.get_shape()[2]])
inputs, encoder_output = build_transformer(net, num_layers, num_heads, ff_dim, dropout_rate)
fully_connected_layer = tflearn.fully_connected(encoder_output, hidden_dim * inputs.get_shape()[1], activation='relu')

# Initialize the TF session and perform training/inference as needed.



# Reshape the output back to 3D
output_reshaped = tflearn.reshape(fully_connected_layer, [-1, 17, 32])

# Create your model by specifying the input layer and output layer
model = tflearn.DNN(output_reshaped) #, optimizer='adam', loss='mean_square')

#model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Prepare your data, for example:
x_train = np.random.randn(17, 3, 1)
y_train = np.random.randn(1, 544)

# Train the model
model.fit(x_train, y_train, batch_size=1)
