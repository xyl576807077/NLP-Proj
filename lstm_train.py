
# coding: utf-8



import pandas as pd
import numpy as np
raw_data = pd.read_csv('./data/sentiment_word_tagging_train.csv', header=None, delimiter='\t')
str(raw_data.iloc[0].values[0])




s = ''
for index, row in raw_data.iterrows():
    if index != 0:
        s = s + ' '
    s = s + row.values[0]




import re
sentences = re.split(u'[，。！？、‘’“”]/[BMENS]', s)
def get_Xy(sentence):
    """将 sentence 处理成 [word1, w2, ..wn], [tag1, t2, ...tn]"""
    sentence = sentence.replace("//", '$/')
    words_tags = re.findall('(.)/(.)', sentence)
    if words_tags:
        words_tags = np.asarray(words_tags)
        words = words_tags[:, 0]
        tags = words_tags[:, 1]
        return words, tags # 所有的字和tag分别存为 data / label
    return None

datas = []
labels = []
for sentence in iter(sentences):
    res = get_Xy(sentence)
    if res:
        datas.append(res[0])
        labels.append(res[1])




df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))

df_data['sentence_len'] = df_data['words'].apply(lambda words: len(words))



from itertools import chain
all_words = list(chain(*df_data['words'].values))





from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(lower=False)
tokenizer.fit_on_texts(all_words)




from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
tag_dict = {'B':1, 'M':2, 'E':3, 'S':4, 'N':5}

def word_to_index(sentence, flag):
    res = []
    if flag == 'content':
        tmp = tokenizer.texts_to_sequences(sentence)
        for i in tmp:
            if i:
                res.append(i[0])
    else:
        for word in sentence:
            res.append(tag_dict[word])
    return [res]







df_data['X'] = df_data['words'].apply(word_to_index, args = ['content'])
df_data['Y'] = df_data['tags'].apply(word_to_index, args = ['sentiment'])




df_data['X'] = df_data['X'].apply(pad_sequences, args=[40, 'int32', 'post'])
df_data['Y'] = df_data['Y'].apply(pad_sequences, args=[40, 'int32', 'post'])




X = np.asarray(list(df_data['X'].values))
y = np.asarray(list(df_data['Y'].values))







X = X.reshape(X.shape[0], X.shape[2])
y = y.reshape(y.shape[0], y.shape[2])




#将标签向量one-hot
def getY(y):
    res = []
    for row in y:
        tmp = []
        for col in row:
            tmp.append(np_utils.to_categorical(col, 6))
        res.append(tmp)
    return np.array(res)
y = getY(y)



y = y.reshape(-1, 40, 6)




maxlen = 40
word_size = 128
class_weight = {0:0.001, 1:0.8, 2:0.8, 3:0.8, 4:0.8, 5:0.2}
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional
from keras.models import Model

sequence = Input(shape=(maxlen,), dtype='int32')
embedded = Embedding(len(all_words)+1, word_size, input_length=maxlen, mask_zero=True)(sequence)
blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded)
output = TimeDistributed(Dense(6, activation='softmax'))(blstm)
model = Model(input=sequence, output=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 1024
history = model.fit(X, y, batch_size=batch_size, epochs=10)
model.save("sentiment_model.hdf5")

