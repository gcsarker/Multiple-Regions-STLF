import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow import keras
from tensorflow.keras import layers

regions = ['dhaka', 'chittagong', 'comilla', 'mymensingh',
       'sylhet', 'khulna', 'rajshahi', 'barishal', 'rangpur']

n_regions = len(regions)  # This corresponds to the number of output heads

def transformer_encoder(inputs, head_size, n_heads, fcl_dim, dropout):
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=n_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    residual = x + inputs

    x = layers.Dense(fcl_dim, activation = layers.LeakyReLU())(residual)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1], activation = layers.LeakyReLU())(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x+residual

def build_transformer_model(input_shape, head_size, n_heads, fcl_dim, n_transformer, dropout):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(n_transformer):
        x = transformer_encoder(x, head_size, n_heads, fcl_dim, dropout)
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = [layers.Dense(1, name = regions[i])(x) for i in range(n_regions)]
    return keras.Model(inputs, outputs)

def build_transformer_lstm_model(input_shape, head_size, n_heads, fcl_dim, n_transformer, dropout):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(n_transformer):
        x = transformer_encoder(x, head_size, n_heads, fcl_dim, dropout)

    #x = layers.LSTM(64, activation = 'tanh', dropout = dropout, return_sequences=True)(x)
    x = layers.LSTM(32, activation = 'tanh', dropout = dropout)(x)
    x = layers.Dense(20, activation="relu")(x)
    #x = layers.Dropout(dropout)(x)
    #x = layers.Dense(20, activation='relu')(x)
    outputs = [layers.Dense(1, name = regions[i])(x) for i in range(n_regions)]
    return keras.Model(inputs, outputs)

def build_cnn_transformer_model(input_shape, head_size, n_heads, fcl_dim, n_transformer, dropout=0):
    inputs = keras.Input(shape=input_shape)
    
    x = inputs
    x = layers.Conv1D(64, 3, activation=layers.LeakyReLU())(x)
    x = layers.MaxPooling1D(2)(x)
    #x = keras.backend.permute_dimensions(x,(0,2,1))
    for _ in range(n_transformer):
        x = transformer_encoder(x, head_size, n_heads, fcl_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    x = layers.Dense(32, activation=layers.LeakyReLU())(x)
    x = layers.Dropout(dropout)(x)
    outputs = [layers.Dense(1, name = regions[i])(x) for i in range(n_regions)]
    return keras.Model(inputs, outputs)


def build_lstm_model(
    input_shape,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    x = layers.LSTM(128, activation = 'tanh',return_sequences=True)(x)
    x = layers.LSTM(64, activation = 'tanh')(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(mlp_dropout)(x)
    outputs = [layers.Dense(1, name = regions[i])(x) for i in range(n_regions)]
    return keras.Model(inputs, outputs)

def build_gru_model(
    input_shape,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    x = layers.GRU(128, activation = 'tanh',return_sequences=True)(x)
    x = layers.GRU(64, activation = 'tanh')(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(mlp_dropout)(x)
    outputs = [layers.Dense(1, name = regions[i])(x) for i in range(n_regions)]
    return keras.Model(inputs, outputs)


def build_cnn_gru_model(
    input_shape,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    x = layers.Conv1D(128, 3, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    #x = layers.GRU(64, activation = 'tanh',return_sequences=True)(x)
    x = layers.GRU(64, activation = 'tanh')(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(mlp_dropout)(x)
    outputs = [layers.Dense(1, name = regions[i])(x) for i in range(n_regions)]
    return keras.Model(inputs, outputs)

def build_cnn_lstm_model(
    input_shape,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    x = layers.Conv1D(128, 3, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LSTM(64, activation = 'tanh',return_sequences=True)(x)
    x = layers.LSTM(64, activation = 'tanh')(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(mlp_dropout)(x)
    outputs = [layers.Dense(1, name = regions[i])(x) for i in range(n_regions)]
    return keras.Model(inputs, outputs)
