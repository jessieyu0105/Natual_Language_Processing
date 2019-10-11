import sys
from collections import defaultdict,Counter
import math
import random
import os
import os.path


def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):

    if n == 1 :
        start_tag = ['START']
    else:
        start_tag = ['START' for _ in range(n-1)]
    sequence = start_tag + sequence
    sequence.append('STOP')
    ngram = [tuple(sequence[i:n+i]) for i in range(len(sequence)-n+1)]
    return ngram


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)
        self.sum_of_unigram_count = sum(self.unigramcounts.values())
        self.sentence_count = self.unigramcounts[('START',)]  # number of sentences
        # Since "START" is appended to the beginning of each sentence in the corpus, it should represent the number of sentences in the training corpus

    def count_ngrams(self, corpus):

        self.unigramcounts = defaultdict(int) 
        self.bigramcounts = defaultdict(int) 
        self.trigramcounts = defaultdict(int)

        ngrams = [self.unigramcounts,self.bigramcounts,self.trigramcounts]
        for sentence in corpus:
            for i,ngram in enumerate(ngrams):
                ng_count = Counter(get_ngrams(sentence,i+1))
                for key in ng_count:
                    ngram[key] += ng_count[key]

        

    def raw_trigram_probability(self,trigram):
    
        a,b,c = trigram   # P(c|(a,b))
        
        # If trigram begins with ("START", "START"), we assume the bigram ("START", "START") to be equal to the total number of sentences in the corpus
        if trigram[:-1] == ("START", "START"):
            return self.trigramcounts[trigram]/self.sentence_count
        else:
            return self.trigramcounts.get(trigram, 0)/self.bigramcounts.get((a,b), 10**(-6)) # address the situation if denominator = 0


    def raw_bigram_probability(self, bigram):

        a,b = bigram   # P(b|a)
        return self.bigramcounts.get(bigram, 0)/self.unigramcounts.get((a,), 10**(-6))
        
        
    
    def raw_unigram_probability(self, unigram):

        if unigram in self.unigramcounts.keys():
            return self.unigramcounts[unigram]/self.sum_of_unigram_count
        else:
            return 0.0
          

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        a,b,c = trigram
        return lambda1 * self.raw_trigram_probability(trigram) + lambda2 * self.raw_bigram_probability((b,c)) + lambda3 * self.raw_unigram_probability((c,))
        
    def sentence_logprob(self, sentence):

        prob = 0
        trigrams = get_ngrams(sentence,3)
        for trigram in trigrams :
            prob += math.log2(self.smoothed_trigram_probability(trigram))
        return prob

    def perplexity(self, corpus):

        M = 0 
        sum_prob = 0
        for sentence in corpus:
            M += len(sentence) + 2 # Also count 'START' and 'STOP' for each sentance
            sum_prob += self.sentence_logprob(sentence)
        l = sum_prob/M
        return 2**(-l)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if pp1 < pp2:
                correct += 1
            total += 1
    
        for f in os.listdir(testdir2):
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            if pp2 < pp1:
                correct += 1
            total += 1
        
        return correct/total

if __name__ == "__main__":
    print('Training with dataset '+ str(sys.argv[1])+'.....')
    model = TrigramModel(sys.argv[1]) 
     
    #Testing perplexity: 
    print("Calculating the perplexity for dataset "+ str(sys.argv[2])+'.....')
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)
    
    # Essay scoring experiment: 
    print()
    print("Essay scoring experiment")
    acc = essay_scoring_experiment('train_high.txt', "train_low.txt", "test_high", "test_low")
    print("Accuracy: ",acc)
