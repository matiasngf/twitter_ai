import tensorflow as tf
from tensorflow.python.keras import layers
import gensim
from tensorflow.python import keras
from tensorflow.python.keras import backend as K
import re
import numpy as np
import pandas as pd
import sklearn.metrics
from nltk.corpus import stopwords


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads
        })
        return config
        
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output
    
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        self.att = MultiHeadSelfAttention(embed_dim, num_heads, **kwargs)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
        })
        return config

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TweetToVect():
    encoder_model = False
    word_model = False
    tweet_length = False
    word_dim = False
    
    def _del_repeated(self, char, s):
        return re.sub(rf'\{char}[{char} ]*', ' '+char+' ', s)

    def _space_emogis(self, s):
        return re.sub("(["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "])", r' \1 ', s, flags=re.UNICODE)

    def clean_text(self, tweet):
        
        """Clean and space tweet"""
        s = self._space_emogis(tweet.lower())
        s = re.sub('\n', ' ', s)
        for i in '?!@#':
            s = self._del_repeated(i, s)
        s = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]+\.[a-z]+\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', 'URL', s)
        s = re.sub(r'(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?', 'URL', s)
        regrex_filter = re.compile(pattern = "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\w+"
            u"?!@#"
        "]+", flags = re.UNICODE)
        
        s = regrex_filter.findall(s)
        s = filter(lambda x: (x not in stopwords.words('spanish') or (x in ['si', 'no', 'sí'])), s)
        s = ' '.join(s)
        
        s = re.sub(r'[ ]+', ' ', s)
        return s

    def similar_from_target_vector(self, to_compare_vector, tweet_vectors):
        """x=target vector y=vectors to compare"""
        similarity = sklearn.metrics.pairwise.cosine_similarity(to_compare_vector, tweet_vectors)
        similarity_index = np.argsort(similarity[0])[::-1]
        similarity_sorted = np.sort(similarity[0])[::-1]
        return similarity_index, similarity_sorted

    def get_word_vectors(self, tweet):
        tokenize = lambda text: text.split(' ')
        arr = [ self.word_model.get_vector(w) for w in tokenize(tweet) ]
        arr = arr+arr
        arr_inv = arr[::-1]
        arr = np.array(arr)
        arr_inv = np.array(arr_inv)
        if(len(arr)==self.tweet_length):
            return [arr, arr_inv]
        elif(len(arr)<self.tweet_length):
            to_add = np.zeros((self.tweet_length - len(arr), self.word_dim))
            return [np.concatenate([arr, to_add]), np.concatenate([arr_inv, to_add])]
        else:
            return [arr[0:self.tweet_length], arr_inv[0:self.tweet_length]]
        
    def generate_input(self, vectors_list):
        vectors = np.array([tweet[0].tolist() for tweet in vectors_list])
        vectors_inv = np.array([tweet[1].tolist() for tweet in vectors_list])
        return({
            'vectors': vectors,
            'vectors_inv': vectors_inv
        })
    
    def get_vector(self, tweet, cleaned=False):
        """Get vectors of tweet"""
        if(cleaned==False):
            tweet = self.clean_text(tweet)
        word_vectors = self.get_word_vectors(tweet) #list of vectors
        input_vectors = self.generate_input([word_vectors]) #{vectors, vectors_inv}
        encoded, residual = self.encoder_model.predict(input_vectors)
        return encoded[0]
    
    def get_vectors(self, tweets, cleaned=False):
        """Get vectors of list of tweets"""
        if(cleaned==False):
            tweets = [self.clean_text(i) for i in tweets]
        word_vectors = [self.get_word_vectors(i) for i in tweets] #list of vectors
        input_vectors = self.generate_input(word_vectors) #{vectors, vectors_inv}
        encoded, residual = self.encoder_model.predict(input_vectors)
        return encoded
            
    def load_word_model(self, path=False, model=False):
        """Load WordToVect model"""
        if(not(self.encoder_model and self.word_dim)):
            raise Exception('First load the encoder.')
        if(path):
            model = gensim.models.fasttext.load_facebook_vectors(path)
        elif(model):
            model = model
        else:
            raise Exception('You have to pass a path or model.')
        if(self.word_dim == len(model.get_vector(' '))):
            self.word_model = model
        else:
            raise Exception('WordToVect dim doesn\'t match encoder input.')
            
    def load_encoder(self, file_name=False, path=False, model=False):
        """Load encoder model"""
        if(path and file_name):
            with open(path+file_name+".json", 'r') as f:
                loaded_model_encoder = tf.keras.models.model_from_json(f.read(), custom_objects={'TransformerBlock': TransformerBlock, 'MultiHeadSelfAttention': MultiHeadSelfAttention})
            loaded_model_encoder.load_weights(path+file_name+".h5")
            self.encoder_model = loaded_model_encoder
        elif(model!=False):
            self.encoder_model = model
        else:
            raise Exception('You have to pass a path or model.')
        self.tweet_length = self.encoder_model.input['vectors'].shape[1]
        self.word_dim = self.encoder_model.input['vectors'].shape[2]