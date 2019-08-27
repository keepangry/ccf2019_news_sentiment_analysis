from custom_layers import base_embed_lstm_net, base_embed_cnn_lstm_net, base_multi_channel_net, base_attention_lstm
import keras


def mr_base_model(dataset, vocabulary_size, maxlen, method="base_multi_channel_net", ):
    if method == "base_multi_channel_net":
        model = base_multi_channel_net(vocabulary_size, time_steps=maxlen)
    elif method == "base_attention_lstm":
        model = base_attention_lstm(vocabulary_size, time_steps=maxlen)
    elif method == "base_embed_cnn_lstm_net":
        model = base_embed_cnn_lstm_net(vocabulary_size)
    else:
        model = base_embed_lstm_net(vocabulary_size)

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='auto'
        )
    ]
    model.summary()
    history = model.fit(dataset.x_train, dataset.y_train,
                        epochs=20,
                        batch_size=128,
                        validation_data=(dataset.x_val, dataset.y_val),
                        callbacks=callbacks,
                        verbose=2)
    return model, history

