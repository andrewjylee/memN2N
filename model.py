from data import Data
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def position_encoding(d, J):
    # TODO
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
    def __init__(self, mem_size, vocab_size, sentence_size, batch_size):
        self.dim = 10 # d = edim = 'internal state dimension' 
        self.mem_size = mem_size
        self.vocab_size = vocab_size
        self.sentence_size = sentence_size
        self.batch_size = batch_size
        self.qa_pair_num = qa_pair_num = 5

        self.setup_network()
        self.inference = self._inference()

        _test_op = tf.Print(self.questions, [self.questions, self.B])
        self.test_op = _test_op

    def setup_network(self):
        # Inputs (Placeholders)
        self.sentences = tf.placeholder(tf.int32, [None, self.mem_size, self.sentence_size])
        self.questions = tf.placeholder(tf.int32, [None, self.sentence_size])

        # Internal States (Variables)
        self.A = tf.Variable(tf.random_normal([self.vocab_size, self.dim]))
        self.B = tf.Variable(tf.random_normal([self.vocab_size, self.dim]))

        # TODO: Check sizes
        #self.TA = tf.Variable(tf.random_normal([self.mem_size, self.dim]) # Check sizing later
        #self.C = tf.Variable(tf.random_normal([self.dim, self.vocab_size]) # Check size
        #self.W = tf.Variable(tf.random_normal([self.vocab_size, self.dim])

    def _inference(self):#, placeholder, size_of_h1, size_of_h2):
        #m = tf.nn.embedding_lookup(self.A, self.sentences)
        
        q_embedding = tf.nn.embedding_lookup(self.B, self.questions)
        m_embedding = tf.nn.embedding_lookup(self.A, self.sentences)
        # TODO: Position Encoding
        u = tf.reduce_sum(q_embedding, 1)
        return q_embedding, u

        #u = tf.reduce_sum(q_embedding, 1)
        #u = tf.reduce_sum(self.B * position_encoding(self.dim, self.v), 1)
        #print 'q embedding size and u size = ', q_embedding.get_shape(), u.get_shape()
        #return self.B, q_embedding, u

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
    data = Data("/Volumes/Data/research/sandbox/data/tasks_1-20_v1-2/en/", 1, 30)
    data.run()
    #data.print_summary()

    stories_train, stories_val, questions_train, questions_val, answers_train, answers_val = train_test_split(data.texts, data.questions, data.answers, test_size=0.1) 
    batch_size = 20

    s = stories_train[0:batch_size]
    q = questions_train[0:batch_size]
    a = answers_train[0:batch_size]

    # Flatten lists out
    print 'before:'
    print s
    print len(s)
    q = [question for sublist in q for question in sublist]
    s = [sentence for sublist in s for sentence in sublist]
    print 'after'
    print s
    print len(s)
    print len(s)

    ###############
    # Build Model
    ###############
    model = memN2N(data.mem_size, data.vocab_size, data.max_sentence_size, batch_size)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    ###############
    # Train
    ###############
    for i in range(0, 1):
        feed_dict = {model.sentences: s, model.questions: q}
        #print sess.run(model.test_op, feed_dict=feed_dict)
        tmp = sess.run(model.inference, feed_dict=feed_dict)
        print tmp, tmp.shape
    sess.close()
