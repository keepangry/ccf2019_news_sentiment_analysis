from keras import layers, Input, regularizers
from keras.models import Sequential, Model
from attention_lstm import attention_3d_block
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from keras.layers import multiply


def base_attention_lstm(vocabulary_size, time_steps=32, type="after"):
    """

    :param vocabulary_size:
    :param time_steps:
    :param type: before/after
    :return:
    """
    INPUT_DIM = 64
    LSTM_UNITS = 32

    text_input = Input(shape=(time_steps,), dtype='int32', name='text')     # (batch_size, time_steps)
    inputs = layers.Embedding(vocabulary_size, INPUT_DIM)(text_input)       # (batch_size, time_steps, input_dim)

    if type == "before":
        output_attention_mul = attention_3d_block(inputs, time_steps=time_steps, single_attention_vector=True)
        attention_mul = LSTM(LSTM_UNITS, return_sequences=False)(output_attention_mul)
    else:   # after
        lstm_out = LSTM(LSTM_UNITS, return_sequences=True)(inputs)
        lstm_out = Reshape((time_steps, LSTM_UNITS))(lstm_out)
        attention_mul = attention_3d_block(lstm_out, time_steps=time_steps, single_attention_vector=True)
        attention_mul = Flatten()(attention_mul)
        attention_mul = Dense(64)(attention_mul)

    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=text_input, outputs=output)

    return model


def base_embed_lstm_net(vocabulary_size):
    model = Sequential()
    model.add(layers.Embedding(vocabulary_size, 128))
    # 0.95
    # model.add(layers.LSTM(64))

    # 0.958
    # model.add(layers.Bidirectional(layers.LSTM(64)))

    # 0.9817
    model.add(layers.Bidirectional(layers.LSTM(64, dropout=0.1, recurrent_dropout=0.5, return_sequences=True)))
    model.add(layers.Bidirectional(layers.LSTM(64, dropout=0.1, recurrent_dropout=0.5)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


def base_embed_cnn_lstm_net(vocabulary_size):
    """

    :param vocabulary_size:
    :return:
    """
    model = Sequential()
    model.add(layers.Embedding(vocabulary_size, 64))
    model.add(layers.Conv1D(64, 4, padding='valid', activation='relu', strides=1))
    model.add(layers.MaxPooling1D(pool_size=3))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


def base_multi_channel_net__runable(vocabulary_size):
    """
    可执行的多路一维卷积
    :param vocabulary_size:
    :return:
    """
    text_input = Input(shape=(32, ), dtype='int32', name='text')
    embedded_text = layers.Embedding(vocabulary_size, 64)(text_input)

    # kernel_size = 3
    channel1 = layers.Conv1D(64, 3, padding="same", activation='relu')(embedded_text)
    channel1 = layers.MaxPool1D(4)(channel1)

    channel2 = layers.Conv1D(64, 4, padding="same", activation='relu')(embedded_text)
    channel2 = layers.MaxPool1D(4)(channel2)

    channel3 = layers.Conv1D(64, 5, padding="same", activation='relu')(embedded_text)
    channel3 = layers.MaxPool1D(4)(channel3)

    concatenated = layers.concatenate([channel1, channel2, channel3], axis=-1)
    concatenated = layers.LSTM(64)(concatenated)

    output = layers.Dense(64, activation='relu')(concatenated)
    output = layers.Dense(1, activation='sigmoid')(output)
    model = Model(text_input, output)
    return model


def base_multi_channel_net(vocabulary_size, time_steps=32):
    """
    多通道卷积
    :param vocabulary_size:
    :param time_steps:
    :return:
    """
    text_input = Input(shape=(time_steps, ), dtype='int32', name='text')
    embedded_text = layers.Embedding(vocabulary_size, 128)(text_input)

    channels = []
    for kernel_size in range(3, 7):
        channel = layers.Conv1D(16, kernel_size, activation='relu')(embedded_text)
        channel = layers.MaxPool1D(time_steps-kernel_size+1)(channel)
        channels.append(channel)

    concatenated = layers.concatenate(channels, axis=-1)
    concatenated = layers.Flatten()(concatenated)
    # concatenated = layers.LSTM(64)(concatenated)
    output = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(concatenated)
    output = layers.Dense(1, activation='sigmoid')(output)
    model = Model(text_input, output)
    return model


def base_multi_channel_softmax_net(vocabulary_size, time_steps=32):
    """
    多通道卷积
    :param vocabulary_size:
    :param time_steps:
    :return:
    """
    text_input = Input(shape=(time_steps, ), dtype='int32', name='text')
    embedded_text = layers.Embedding(vocabulary_size, 128)(text_input)

    channels = []
    for kernel_size in range(3, 7):
        channel = layers.Conv1D(16, kernel_size, activation='relu')(embedded_text)
        channel = layers.MaxPool1D(time_steps-kernel_size+1)(channel)
        channels.append(channel)

    concatenated = layers.concatenate(channels, axis=-1)
    concatenated = layers.Flatten()(concatenated)
    # concatenated = layers.LSTM(64)(concatenated)
    output = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(concatenated)
    output = layers.Dense(3, activation='softmax')(output)
    model = Model(text_input, output)
    return model

def base_wide_deep_net(vocabulary_size):
    pass


if __name__ == "__main__":
    # model = base_multi_channel_net(10000)
    model = base_attention_lstm(10000)
    model.summary()

