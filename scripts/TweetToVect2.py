import tensorflow as tf
import clean_text
import gensim.models.fasttext
import numpy as np

class TweetToVect():
    word_model = False
    word_dim = False
    
    def clean_string(self, string, filter_stopwords=False):
        return clean_text.clean(string, filter_stopwords)

    def load_model(self, model_path=False, fast_text_model=False):
        """Load FastText model"""
        if((not model_path) & (not fast_text_model)):
            raise Exception('You have to pass a path or model.')
        elif(model_path):
            model = gensim.models.fasttext.load_facebook_vectors(model_path)
        else:
            model = fast_text_model
        self.word_dim = len(model.get_vector('test'))
        self.word_model = model
        
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
        

    def __sentence_emb_to_vect(self, word_vectors):
        vectors = tf.constant(word_vectors)
        return tf.math.reduce_mean(vectors, axis=1).numpy()
            
    def get_word_vectors(self, tweet):
        tokenize = lambda text: text.split(' ')
        arr = [ self.word_model.get_vector(w) for w in tokenize(tweet) ]
        return arr
    
    def sent_emb_to_vect(self, tweets, return_tensor=False):
        tweets = [tf.reduce_mean(tf.constant(np.transpose(i)), axis=1).numpy() for i in tweets]
        tweets = tf.constant(tweets)
        if(return_tensor):
            return tweets
        else:
            return tweets.numpy()

    def get_similar_word(self, positive=None, negative=None, topn=10):
        return self.word_model.wv.most_similar(positive, negative, topn)

    def get_vector(self, tweet, cleaned=False, return_tensor=False):
        """Get vectors of tweet"""
        if(cleaned==False):
            tweet = self.clean_string(tweet)
        word_vectors = self.get_word_vectors(tweet) #tweet vector
        return self.sent_emb_to_vect([word_vectors], return_tensor)[0]
            
    def get_vectors(self, tweets, cleaned=False, return_tensor=False):
        """Get vectors of list of tweets"""
        if(cleaned==False):
            tweets = [self.clean_string(i) for i in tweets]
        word_vectors = [self.get_word_vectors(i) for i in tweets] #list of vectors
        return self.sent_emb_to_vect(word_vectors, return_tensor)