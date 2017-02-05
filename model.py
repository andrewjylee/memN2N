from data import Data
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

#data = Data("/Volumes/Data/research/sandbox/data/tasks_1-20_v1-2/en/", 1)
#data.run()

#print(data.data[-1][0])
#vocab_size = len(data.word2idx)

#print data.max_sentence_size

#p = tf.nn.softmax(logits)

#texts = tf.placeholder(tf.int32, shape=())
#feed_dict = {texts: }
#sess.run([], feed_dict=feed_dict)


def position_encoding(d, J):
    # Position Encoding described in section 4.1 [1]
    # d = dimension of embedding
    # J = vocab_size
    encoding = np.ones((d, J), dtype=np.float32)
    ls = d+1
    le = J+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / d / J
    return np.transpose(encoding)

   


# Notes:
#tf.nn.embedding_lookup(params, ids) = Ax

class memN2N(object):
    def __init__(self, vocab_size):
        self.dim = 10 # d = edim = 'internal state dimension' 
        self.mem_size = 100
        self.vocab_size = vocab_size
        self.question_size = 3
        # TOOD: fix 3 to appropriate size - need to pad questions with 0s
        self.questions = tf.placeholder(tf.int32, [self.question_size, None])
        self.B = tf.Variable(tf.random_normal([self.dim, self.vocab_size]))
        #self.B = tf.Variable(tf.random_normal([self.vocab_size, self.dim]))

        _test_op = tf.Print(self.questions, [self.questions])

        self.test_op = _test_op
        self.inference = self._inference()
        # TODO: Check sizes
        #self.A = tf.Variable(tf.random_normal([self.dim, self.vocab_size]) #[vocab_size, dim]? 
        #self.TA = tf.Variable(tf.random_normal([self.mem_size, self.dim]) # Check sizing later
        #self.C = tf.Variable(tf.random_normal([self.dim, self.vocab_size]) # Check size
        #self.W = tf.Variable(tf.random_normal([self.vocab_size, self.dim])

    def _inference(self):#, placeholder, size_of_h1, size_of_h2):
        tf.reshape(self.questions, [1, 3])
        q_embedding = tf.nn.embedding_lookup(self.B, self.questions)

        u = tf.reduce_sum(q_embedding, 1)
        #u = tf.reduce_sum(self.B * position_encoding(self.dim, self.v), 1)
        #print 'q embedding size and u size = ', q_embedding.get_shape(), u.get_shape()
        return self.B, q_embedding, u

        #print q_embedding

        #m = tf.nn.embedding_lookup(self.A, self.stories)
        #m = tf.reduce_sum(m * position_encoding(self.dim, self.vocab_size), 1)
        #c = 
        #u = 
        #p_i = tf.nn.log_softmax(uTm_i)
        #c_i = 

        #W = 
        #o = tf.reduce_sum(tf.multiply(p_i, c_i))
        #logits = tf.matmul(W, tf.add(o, u), name='logits')
        #answer = tf.nn.log_softmax(logits, name='Final Answer')

if __name__ == '__main__':
    ###############
    # Data
    ###############
    data = Data("/Volumes/Data/research/sandbox/data/tasks_1-20_v1-2/en/", 1)
    data.run()
    #data.print_summary()

    stories_train, stories_val, questions_train, questions_val, answers_train, answers_val = train_test_split(data.texts, data.questions, data.answers, test_size=0.1) 

    ###############
    # Build Model
    ###############
    model = memN2N(self.vocab_size)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

'''
    ###############
    # Train
    ###############
    for i in range(0, 2):
        print data.questions[i]
        print 'feed dict model.questions size: ', np.transpose(np.array(data.questions[i], ndmin=2)).shape
        feed_dict = {model.questions: np.transpose(np.array(data.questions[i], ndmin=2))}
        print sess.run(model.inference, feed_dict=feed_dict)
    sess.close()
'''
