from data import Data
from model import memN2N
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


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

