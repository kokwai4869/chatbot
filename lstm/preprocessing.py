import nltk
import numpy as np

from itertools import chain
from nltk.tokenize import word_tokenize
from keras.preprocessing.sequence import pad_sequences

class preprocessedInputOutput(object):
    def __init__(self, encoder_input: list, decoder_input: list,
                 padding='post', truncating='post'):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.sentence_tokenizer()
        self.sentence_sequence()
        self.sentence_padding(padding, truncating)
        self.sentence_onehot()
        
    def sentence_tokenizer(self):
        self.tokenized_enc = [word_tokenize(question) for
                              question in self.encoder_input]
        self.tokenized_dec = [word_tokenize(question) for
                              question in self.decoder_input]
        
    def sentence_sequence(self):
        total_words = set(list(chain.from_iterable(self.tokenized_enc)) +\
                          list(chain.from_iterable(self.tokenized_dec)))
        self.word_2_int = {word:i for i, word in enumerate(total_words)}
        self.int_2_word = {i:word for word, i in self.word_2_int.items()}
        self.sequence_enc = [[self.word_2_int[word] for word in sentence] for
                             sentence in self.tokenized_enc]
        self.sequence_dec = [[self.word_2_int[word] for word in sentence] for
                             sentence in self.tokenized_dec]

    def sentence_padding(self, padding, truncating):
        len_enc = [len(seq) for seq in self.sequence_enc]
        len_dec = [len(seq) for seq in self.sequence_dec]
        num_tokens = len_enc + len_dec
        num_tokens = np.array(num_tokens)
        self.sentence_len = int(np.mean(num_tokens) + 2*np.std(num_tokens))
        self.padded_enc = pad_sequences(self.sequence_enc,
                                        maxlen=self.sentence_len,
                                        padding=padding,
                                        truncating=truncating)
        self.padded_dec = pad_sequences(self.sequence_dec,
                                        maxlen=self.sentence_len,
                                        padding=padding,
                                        truncating=truncating)
        
    def sentence_onehot(self):
        def generate_onehot(word, word_dict):
            z = np.zeros(len(word_dict))
            z[word] = 1
            return z

        def get_onehot(series, word_dict):
            onehot_l = []
            for sentence in series:
                onehot_s = [generate_onehot(word, word_dict) for word in sentence]
                onehot_l.append(onehot_s)
            return onehot_l
        self.onehot_dec = np.array(get_onehot(self.padded_dec, self.word_2_int))