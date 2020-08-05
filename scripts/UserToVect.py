import pandas as pd
import numpy as np
import tensorflow as tf
import math
from tensorflow.python.keras import backend as K

class UserToVect():
    num_tweets = False
    tweet_dim = False
    encoder = False
    emb_dim = False
    
    def load_encoder(self, model_path=False, keras_model=False):
        """Load UserToVect model"""
        if((not model_path) & (not keras_model)):
            raise Exception('You have to pass a path or model.')
        elif(model_path):
            with open(model_path+".json", 'r') as f:
                model = tf.keras.models.model_from_json(f.read())
            model.load_weights(model_path+".h5")
        else:
            model = keras_model
        self.num_tweets = model.input.shape[1]
        self.tweet_dim = model.input.shape[2]
        self.emb_dim = model.output[0].shape[1]
        self.encoder = model
        
    def pairwise_distance_vectors(self, x, y):
        """compare two vectors"""
        x = tf.nn.l2_normalize(x, axis = 1)
        y = tf.nn.l2_normalize(y, axis = 1)
        return 1 - tf.matmul(x, y, adjoint_b = True ).numpy()
        
    def pairwise_similarity_vectors(self, x, y):
        """compare two vectors"""
        x = tf.nn.l2_normalize(x, axis = 1)
        y = tf.nn.l2_normalize(y, axis = 1)
        return tf.matmul(x, y, adjoint_b = True ).numpy()
    
    def _sample_encoding(self, mu, log_var):
        z_mean, z_log_var = mu, log_var
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    def _get_user_vector(self, tweet_vectors):
        """Input shape: (None, num_tweets, tweet_dim)"""
        mu, log_var, res = self.encoder(np.array(tweet_vectors))
        embeddings = self._sample_encoding(mu, log_var)
        return embeddings.numpy()
    
    def get_vectors(self, users, max_stacks=5):
        if(self.encoder==False):
            raise Exception('Load encoder first! Use load_encoder()')
        users = [self.get_vector(i) for i in users]
        return np.array(users)
    
    def get_vector(self, tweet_vectors, max_stacks=5):
        if(self.encoder==False):
            raise Exception('Load encoder first! Use load_encoder()')
        tweet_vectors = np.array(tweet_vectors.tolist())
        n_tweets = len(tweet_vectors)
        if(n_tweets<self.num_tweets):
            repeat = math.ceil(self.num_tweets / n_tweets)
            tweet_vectors = np.concatenate([tweet_vectors for i in range(repeat)])
            stacks_num = 1
        if(n_tweets >= self.num_tweets*2):
            stacks_num = math.floor(n_tweets/self.num_tweets)
            stacks_num = stacks_num if stacks_num <= max_stacks else max_stacks
        else:
            stacks_num = 1
        stacks = []
        for i in range(stacks_num):
            stacks.append(tweet_vectors[i*self.num_tweets:i*self.num_tweets+self.num_tweets])
        user_embeddings = self._get_user_vector(stacks) #all vectors from user
        return tf.reduce_mean(tf.constant(np.transpose(user_embeddings)), axis=1).numpy()