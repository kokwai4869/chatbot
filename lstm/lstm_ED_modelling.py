import tensorflow as tf

from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, Activation, dot, concatenate
from keras.models import Model
from keras.utils import to_categorical
tf.keras.backend.clear_session()

class buildModel(object):
    def __init__(self, input_length:int, output_length:int,
                 dict_size:int, embedding_cells:int,
                 lstm_layers:int, lstm_cells:int, input_enc,
                 input_dec, output_dec, model_fit=True):
        self.INPUT_LENGTH = input_length
        self.OUTPUT_LENGTH = output_length
        self.dict_size = dict_size
        self.model_layers(embedding_cells, lstm_layers, lstm_cells)
        self.model_attention()
        self.model_build()
        self.model_fit(model_fit, input_enc, input_dec,output_dec)

    def model_layers(self, embedding_cells, lstm_layers, lstm_cells):
        self.input_enc = Input(shape=(self.INPUT_LENGTH,))
        self.input_dec = Input(shape=(self.OUTPUT_LENGTH,))

        encoder = Embedding(self.dict_size, embedding_cells,
                            input_length=self.INPUT_LENGTH,
                            mask_zero=True)(self.input_enc)
        self.encoder = LSTM(lstm_cells,
                            return_sequences=True,
                            unroll=True)(encoder)
        if lstm_layers > 1:
            for _ in range(lstm_layers):
                self.encoder = LSTM(lstm_cells,
                                    return_sequences=True,
                                    unroll=True)(self.encoder)
        self.encoder_last = self.encoder[:,-1,:]


        decoder = Embedding(self.dict_size,
                            embedding_cells,
                            input_length=self.OUTPUT_LENGTH,
                            mask_zero=True)(self.input_dec)
        self.decoder = LSTM(lstm_cells,
                            return_sequences=True,
                            unroll=True)\
        (decoder, initial_state=[self.encoder_last, self.encoder_last])
        if lstm_layers > 1:
            for _ in range(lstm_layers):
                self.decoder = LSTM(lstm_cells,
                                    return_sequences=True,
                                    unroll=True)(self.decoder)
    
    def model_attention(self):
        attention = dot([self.decoder, self.encoder], axes=[2, 2])
        self.attention = Activation('softmax')(attention)
        self.context = dot([self.attention, self.encoder], axes=[2,1])
        self.decoder_combined_context = concatenate([self.context, self.decoder])

        # Has another weight + tanh layer as described in equation (5) of the paper
        output = TimeDistributed(Dense(512, activation="tanh"))(self.decoder_combined_context)
        self.output = TimeDistributed(Dense(self.dict_size, activation="softmax"))(output)
        
    def model_build(self):
        self.model = Model(inputs=[self.input_enc, self.input_dec], outputs=[self.output])
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        self.summary = self.model.summary()
        
    def model_fit(self, model_fit, input_enc, input_dec, output_dec):
        if model_fit:
            self.model.fit(x=[input_enc, input_dec],
                      y=[output_dec],
                      #validation_split=0.05,
                      batch_size=64, epochs=100)