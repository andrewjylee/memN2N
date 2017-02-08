import os
import copy

class Data(object):
    def __init__(self, data_dir, task_id, mem_size):
        self.data_dir = data_dir
        self.task_id = task_id
        self.data = []
        self.word2idx = {}
        self.idx2word = {}
        self.texts = []
        self.mem_size = mem_size
        self.max_sentence_size = 0

    def run(self):
        self.data, self.word2idx = self.read_data()
        self.idx2word = {v: k for k, v in self.word2idx.iteritems()}
        self.texts = [triplets[0] for triplets in self.data]
        self.questions = [triplets[1] for triplets in self.data]
        self.answers = [triplets[2] for triplets in self.data]
        self.vocab_size = len(self.word2idx)

    def read_data(self):
        filename = self.data_dir + "qa" + str(self.task_id) + "_"
        files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir)]

        train_file = [f for f in files if filename in f][1]
        test_file = [f for f in files if filename in f][0]

        if not os.path.isfile(train_file):
            raise Exception("Can't find " + train_file)

        if not os.path.isfile(test_file):
            raise Exception("Can't find " + test_file)

        self.idx2word, self.word2idx = self.build_idx2word(train_file)
        self.idx2word, self.word2idx = self.build_idx2word(test_file)
        #print self.idx2word, self.word2idx

        self.data = self.parse_data(train_file)
        
        self.data = self.vectorize()
        # TODO
        # Parse test_file too?

        return self.data, self.word2idx

    def build_idx2word(self, file_name):
        count = len(self.idx2word)
        if count == 0:
            count = 1
        with open(file_name) as f:
            for line in f.readlines():
                words = line.split(' ')[1:]
                if len(words) > self.max_sentence_size:
                    self.max_sentence_size = len(words)
                for word in words:
                    word = self.clean_word(word)
                    if word not in self.word2idx:
                        self.word2idx[word] = count
                        self.idx2word[count] = word
                        count += 1
        return self.idx2word, self.word2idx

    def parse_data(self, filename):
        # Returns list of [(list of text sentences), question, answer]
        data_triplets = []
        with open(filename) as f:
            for line in f.readlines():
                if int(line.split(' ')[0]) == 1:
                    # Beginning of Story
                    text_set = []
                    intermediate_text_set = []
                    question_set = []
                    answer_set = []
                        
                if '\t' in line:
                    # Question & Answer
                    splits = line.split('\t')
                    question = splits[0].split(' ', 1)[1].split('?')[0]
                    answer = splits[1]
                    # TOOD: Fully understand why deepcopy is needed,
                    # but not needed in vectorize()
                    text_set.append(copy.deepcopy(intermediate_text_set))
                    question_set.append(question)
                    answer_set.append(answer)
                else:
                    # Sentence
                    intermediate_text_set.append([line.split(' ', 1)[1].split('\n')[0]])

                if int(line.split(' ')[0]) == 15:
                    # End of Story
                    data_triplets.append([text_set, question_set, answer_set])
                    
        return data_triplets


    def vectorize(self):
        # Convert list of text, question, answer to vectorized version using word2idx
        vectorized_data = []
        for text_set, questions, answers in self.data:
            vectorized_text = []
            vectorized_question = []
            vectorized_answer = []
            for sentences in text_set:
                #print sentences
                subtext_set = []
                for sentence in sentences:
                    subtext = []
                    for word in sentence[0].split(' '):
                        subtext.append(self.word2idx[self.clean_word(word)])
                    self.pad_sentence(subtext)
                    subtext_set.append(subtext)
                #vectorized_text.append(copy.deepcopy(subtext_set))

                subtext_set = [i for sublist in subtext_set for i in sublist]
                subtext_set = self.pad_to_memory_size(subtext_set)
                vectorized_text.append(subtext_set)
            for question in questions:
                question_set = []
                for word in question.split(' '):
                    question_set.append(self.word2idx[self.clean_word(word)])
                # TODO: pad question
                vectorized_question.append(list(question_set))
            for answer in answers:
                answer_set = []
                for word in answer.split(' '):
                    answer_set.append(self.word2idx[self.clean_word(word)])
                vectorized_answer.append(list(answer_set))
            #vectorized_data.append([list(vectorized_text), list(vectorized_question), list(vectorized_answer)])
            vectorized_data.append([vectorized_text, vectorized_question, vectorized_answer])
        return vectorized_data

    def pad(self):
        for story in self.texts:
            for sentence in story:
                if len(sentence) < self.max_sentence_size:
                    sentence.extend([0 for x in range(self.max_sentence_size - len(sentence))])
        for questions in self.questions:
            for question in questions:
                if len(question) < self.max_sentence_size:
                    question.extend([0 for x in range(self.max_sentence_size - len(question))])
    
    def pad_sentence(self, sentence):
        if len(sentence) < self.max_sentence_size:
            #sentence.extend([0 for x in range(self.max_sentence_size - len(sentence))])
            sentence += [0 for x in range(self.max_sentence_size - len(sentence))]

    def pad_to_memory_size(self, text):
        if len(text) < self.mem_size:
            #text.extend([0 for i in range(self.mem_size - len(text))])
            text += [0 for i in range(self.mem_size - len(text))]
        else:
            # Only save the most recent 'self.max_sentence_size' entries
            text = text[-(self.mem_size-len(text)):]
        assert(len(text) == self.mem_size)
        return text




    def clean_word(self, word):
        if '.' in word:
            word = word.split('.')[0]
        if '\n' in word:
            word = word.split('\n')[0]
        if '\t' in word:
            word = word.split('\t')[1]
        if '?' in word:
            word = word.split('?')[0]
        return word

    def print_summary(self):
        print 'Data Summary: '
        print self.idx2word

        print '  Size of Data:'
        print len(self.texts), len(self.questions), len(self.answers)
        print self.texts[-1]

        print self.questions[-1]

        print self.answers[-1]
