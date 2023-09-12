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
    x = inputs

    for _ in range(num_layers):
        x = transformer_encoder_block(x, num_heads, ff_dim, dropout_rate)

    return inputs, x

# Example usage
input_shape = (17, 3)  # Sequence length and embedding dimension
num_layers = 6
num_heads = 3
ff_dim = 32
dropout_rate = 0.1

inputs, encoder_output = build_transformer_encoder(input_shape, num_layers, num_heads, ff_dim, dropout_rate)

# Initialize a TensorFlow session
with tf.Session() as sess:
    # Create some example data
    #example_data = tf.random.uniform(shape=[64, 17, 3], dtype=tf.float32)
    #example_data = tf.constant(1.0, shape=(64, 17, 3), dtype=tf.float32)  # Example inputs
    # Run the session with the input data
    #output = sess.run(encoder_output, feed_dict={inputs: example_data})

    # Create some example data (as a NumPy array)
    example_data = np.random.randn(64, 17, 3).astype(np.float32)  # Replace with your actual data

    # Run the session with the input data
    output = sess.run(encoder_output, feed_dict={inputs: example_data})

    print(encoder_output.eval())

#with tf.Session() as sess:
#    print(inputs.eval())
#    print(encoder_output.eval())
# Initialize the TF session and perform training/inference as needed.
