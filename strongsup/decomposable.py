from keras.layers import Input, Dense, merge, GlobalAveragePooling1D, GlobalMaxPooling1D, BatchNormalization, Dropout
from keras.layers import Activation, TimeDistributed
from keras.layers import Bidirectional, LSTM
import keras.backend as K
from keras.models import Sequential, Model
from keras.optimizers import Adam, Adagrad
from keras.regularizers import l2


def decomposable_model_generation(utter_shape, path_shape, classifications, settings):
    hidden_len = settings['hidden_layers']
    max_utters, utter_size = utter_shape
    max_paths, path_size = path_shape
    utterance_inp = Input(shape=(max_utters, utter_size), dtype='float32', name='uttr_inp')
    path_inp = Input(shape=(max_paths, path_size), dtype='float32', name='path_inp')

    # Construct operations, which we'll chain together.
    encode_utt = BiLSTM(max_utters, utter_size, hidden_len, dropout=settings['dropout'])
    encode_path = BiLSTM(max_paths, path_size, hidden_len, dropout=settings['dropout'])
    attend = Attention(hidden_len, dropout=settings['dropout'])
    align = Alignment(hidden_len)
    compare = CompareAndAggregate(hidden_len, dropout=settings['dropout'])
    output_score = Ranker(hidden_len, classifications, dropout=settings['dropout'])

    uttr_enc = encode_utt(utterance_inp)
    path_enc = encode_path(path_inp)

    attention = attend(uttr_enc, path_enc, utter_size, path_size)

    align_uttr = align(path_enc, attention, max_utters)
    align_path = align(uttr_enc, attention, max_paths, transpose=True)

    compare_uttr = compare(uttr_enc, align_uttr, utter_size, True)
    compare_path = compare(path_enc, align_path, path_size, False)

    ranks = output_score(compare_uttr, compare_path)

    decomposable_model = Model(input=[utterance_inp, path_inp], output=[ranks])

    if settings['optimizer'].lower() == 'adagrad':
        optimizer = Adagrad(lr=settings['lr'])
    else:
        optimizer = Adam(lr=settings['lr'])

    decomposable_model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy'])

    return decomposable_model


class BiLSTM(object):
    def __init__(self, max_length, token_size, hidden_len, dropout=0.0):
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(hidden_len, return_sequences=True,
                                     dropout_W=dropout, dropout_U=dropout),
                                     input_shape=(max_length, token_size)))
        self.model.add(TimeDistributed(Dense(hidden_len, activation='relu', init='he_normal')))
        self.model.add(TimeDistributed(Dropout(0.2)))

    def __call__(self, sentence):
        return self.model(sentence)


class Attention(object):
    def __init__(self, hidden_len, dropout=0.0, L2=0.0):
        self.model_utter = Sequential()
        self.model_utter.add(Dropout(dropout, input_shape=(hidden_len,)))
        self.model_utter.add(
            Dense(hidden_len, name='attend1',
                  init='he_normal', W_regularizer=l2(L2),
                  input_shape=(hidden_len,),
                  activation='relu'))
        self.model_utter.add(Dropout(dropout))
        self.model_utter.add(Dense(hidden_len, name='attend2',
                                   init='he_normal', W_regularizer=l2(L2), activation='relu'))
        self.model_utter = TimeDistributed(self.model_utter)

        self.model_path = Sequential()
        self.model_path.add(Dropout(dropout, input_shape=(hidden_len,)))
        self.model_path.add(
            Dense(hidden_len, name='attend3',
                  init='he_normal', W_regularizer=l2(L2),
                  input_shape=(hidden_len,),
                  activation='relu'))
        self.model_path.add(Dropout(dropout))
        self.model_path.add(Dense(hidden_len, name='attend4',
                                  init='he_normal', W_regularizer=l2(L2), activation='relu'))
        self.model_path = TimeDistributed(self.model_path)

    def __call__(self, utter, path, max_len_utter, max_len_path):
        # attend step that skips the quadratic complexity of normal attention
        def merge_mode(utter_path):
            bitwise_attention = K.batch_dot(utter_path[1], K.permute_dimensions(utter_path[0], (0, 2, 1)))
            return K.permute_dimensions(bitwise_attention, (0, 2, 1))

        utter_model = self.model_utter(utter)
        path_model = self.model_path(path)

        return merge(
            [utter_model, path_model],
            mode=merge_mode,
            output_shape=(max_len_utter, max_len_path))


class Alignment(object):
    def __init__(self, hidden_len):
        self.hidden_len = hidden_len

    def __call__(self, sentence, attentions_matrix, max_length, transpose=False):
        def normalize_attention(attention_matrix_sentence):
            attention_matrix = attention_matrix_sentence[0]
            att_sentence = attention_matrix_sentence[1]
            if transpose:
                attention_matrix = K.permute_dimensions(attention_matrix, (0, 2, 1))
            # softmax attention
            exp_mat = K.exp(attention_matrix - K.max(attention_matrix, axis=-1, keepdims=True))
            sum_mat = K.sum(exp_mat, axis=-1, keepdims=True)
            softmax_attention = exp_mat / sum_mat
            return K.batch_dot(softmax_attention, att_sentence)

        return merge([attentions_matrix, sentence], mode=normalize_attention,
                     output_shape=(max_length, self.hidden_len))


class CompareAndAggregate(object):
    def __init__(self, hidden_len, L2=0.0, dropout=0.0):
        self.model_utt = Sequential()
        self.model_utt.add(Dropout(dropout, input_shape=(hidden_len * 2,)))
        self.model_utt.add(Dense(hidden_len, name='compare1',
                                 init='he_normal', W_regularizer=l2(L2)))
        self.model_utt.add(Activation('relu'))
        self.model_utt.add(Dropout(dropout))
        self.model_utt.add(Dense(hidden_len, name='compare2',
                                 init='he_normal', W_regularizer=l2(L2)))
        self.model_utt.add(Activation('relu'))
        self.model_utt = TimeDistributed(self.model_utt)

        self.model_path = Sequential()
        self.model_path.add(Dropout(dropout, input_shape=(hidden_len * 2,)))
        self.model_path.add(Dense(hidden_len, name='compare3',
                                  init='he_normal', W_regularizer=l2(L2)))
        self.model_path.add(Activation('relu'))
        self.model_path.add(Dropout(dropout))
        self.model_path.add(Dense(hidden_len, name='compare4',
                                  init='he_normal', W_regularizer=l2(L2)))
        self.model_path.add(Activation('relu'))
        self.model_path = TimeDistributed(self.model_path)

    def __call__(self, sent, align, max_len, is_model_utt, **kwargs):
        if is_model_utt:
            result = self.model_utt(merge([sent, align], mode='concat'))
        else:
            result = self.model_path(merge([sent, align], mode='concat'))
        avged = GlobalAveragePooling1D()(result, mask=max_len)
        maxed = GlobalMaxPooling1D()(result, mask=max_len)
        merged = merge([avged, maxed])
        result = BatchNormalization()(merged)
        return result


class Ranker(object):
    def __init__(self, hidden_len, out_len, dropout=0.0, L2=0.0):
        self.model = Sequential()
        self.model.add(Dropout(dropout, input_shape=(hidden_len * 2,)))
        self.model.add(Dense(hidden_len, name='hidden_rank',
                             init='he_normal', W_regularizer=l2(L2)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(hidden_len, name='entail2',
                             init='he_normal', W_regularizer=l2(L2)))
        self.model.add(Activation('relu'))
        self.model.add(Dense(out_len, name='entail_out', activation='softmax',
                             W_regularizer=l2(L2), init='zero'))

    def __call__(self, compare_utter, compare_path):
        ranker = merge([compare_utter, compare_path], mode='concat')
        ranker = self.model(ranker)
        return ranker
