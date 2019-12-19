import sys

from lexsub_xml import read_lexsub_xml


from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
import numpy as np
import string



def tokenize(s):
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos):
    possible_synonyms = set()
    lemmas = wn.lemmas(lemma, pos=pos)
    for lm in lemmas:
        synset = lm.synset()
        for syn_lemma in synset.lemmas():
            lemma_name = syn_lemma.name()
            if lemma_name.lower() != lemma:
                lemma_name = lemma_name.replace('_', ' ')
                possible_synonyms.add(lemma_name)
    return possible_synonyms


def wn_frequency_predictor(context):
    lemma = context.lemma
    pos = context.pos
    occur_freq = {}
    lemmas = wn.lemmas(lemma, pos=pos)
    for lm in lemmas:
        synset = lm.synset()
        for syn_lemma in synset.lemmas():
            lemma_name = syn_lemma.name()
            if lemma_name.lower() != lemma:
                lemma_name = lemma_name.replace('_', ' ')
                occur_freq[lemma_name] = occur_freq.get(lemma_name, 0) + syn_lemma.count()
    ordered_occur_freq = [(lemma_name, freq) for lemma_name, freq in occur_freq.items()]
    ordered_occur_freq.sort(key=lambda x: x[1], reverse=True)
    return ordered_occur_freq[0][0]



def wn_simple_lesk_predictor(context):
    stop_words = stopwords.words('english')
    wnl = WordNetLemmatizer()
    # Based on context, get lemma, POS, left & right context
    lemma = context.lemma
    pos = context.pos
    left_context = context.left_context
    right_context = context.right_context
    # Deal with the context set
    new_context = ' '.join(left_context + right_context)                                # context without the target word
    new_context_words = tokenize(new_context)                                           # remove punctuations
    new_context_words = [word for word in new_context_words if word not in stop_words]  # remove stop_words
    new_context_words = set(new_context_words)          # convert into set, remove repeating words

    overlaps = []  # For saving synsets, if there is overlaps

    lemmas = wn.lemmas(lemma, pos=pos)
    for lm in lemmas:
        synset = lm.synset()
        # Add: definition and all examples for the synset
        definitions = list()
        definitions.append(synset.definition())
        definitions += synset.examples()
        # Add: definition and all examples for all hypernyms of the synset
        hypernyms = synset.hypernyms()
        if hypernyms:
            for i in range(len(hypernyms)):
                definitions.append(hypernyms[i].definition())
                definitions += hypernyms[i].examples()
        # Deal with the synsets set
        definitions = ' '.join(definitions)
        def_words = tokenize(definitions)                                    # remove punctuations
        def_words = [word for word in def_words if word not in stop_words]   # remove stop_words
        def_words = [wnl.lemmatize(word) for word in def_words]              # lemmatize: I use this and improve the precision
        def_words = set(def_words)                                           # convert into set, remove repeating words
        
        # Take intersection of the two sets
        overlap = new_context_words & def_words
        if len(overlap) != 0:
            overlaps.append(synset)


    # If overlaps existed, iterate, find out the most frequent lemma_name which != target_word (lemma)
    if overlaps:
        lexeme_freqs = {}
        for synset in overlaps:
            for syn_lemma in synset.lemmas():
                lemma_name = syn_lemma.name()
                if lemma_name.lower() != lemma:
                    lemma_name = lemma_name.replace('_', ' ')
                    lexeme_freqs[lemma_name] = lexeme_freqs.get(lemma_name, 0) + 1
        # If such lemma_name existed, sort, and choose the most frequent one
        if len(lexeme_freqs) != 0:
            lexeme_freqs_list = [(lemma_name, freq) for lemma_name, freq in lexeme_freqs.items()]
            lexeme_freqs_list.sort(key=lambda x: x[1], reverse=True)
            return lexeme_freqs_list[0][0]
        # Otherwise, choose the most frequent synset and the most frequent synonym according to WordNet
        else: 
            return wn_frequency_predictor(context)         
    # If NO overlaps existed, choose the most frequent synset and the most frequent synonym according to WordNet
    else:
        return wn_frequency_predictor(context)
       
   
class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context):
        lemma = context.lemma
        pos = context.pos
        # Obtain all possible synonyms
        possible_synonyms = get_candidates(lemma, pos)
        
        similarities = []
        # Iterate over possible synonyms，calculate similarities
        for p_synonym in possible_synonyms:
            p_synonym_restore = p_synonym.replace(' ', '_')
            # Check if the lemme belongs to the model
            if p_synonym_restore in self.model.vocab:
                similarity = self.model.similarity(lemma, p_synonym_restore)
            else:
                similarity = 0
            similarities.append((p_synonym, similarity))
        # Order by similarities
        similarities.sort(key=lambda x: x[1], reverse=True)
        # If there exists possible synonyms，return the one with highest similarity
        if similarities:
            nearest = similarities[0][0]
            return nearest
        else:
            return ""


    # Define a function cos(v1, v2) for calculating cosine similarity later
    def cos(self, v1, v2):
        return np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))

    def predict_nearest_with_context(self, context): 
        stop_words = stopwords.words('english')
        # Based on context, get lemma, POS, left & right context
        lemma = context.lemma
        pos = context.pos
        left_context = context.left_context
        right_context = context.right_context
        
        new_context = left_context[-5:] + right_context[:5] # limit the context to +-5 words
        new_context = [word for word in new_context if word not in stop_words]  # remove stop_words
        
        # word vector
        sen_vec = self.model.wv[lemma].copy()
        for word in new_context:
            if word in self.model.vocab:
                sen_vec += self.model.wv[word]

        # Obtain all possible synonyms
        possible_synonyms = get_candidates(lemma, pos)
        
        similarities = []
        # Iterate over possible synonyms，calculate similarities
        for p_synonym in possible_synonyms:
            p_synonym_restore = p_synonym.replace(' ', '_')
            # Check if the lemme belongs to the model
            if p_synonym_restore in self.model.vocab:
                similarity = self.cos(self.model.wv[p_synonym_restore], sen_vec)
            else:
                similarity = 0
            similarities.append((p_synonym, similarity))
        # Order by similarities
        similarities.sort(key=lambda x: x[1], reverse=True)
        # If there exists possible synonyms，return the one with highest similarity
        if similarities:
            nearest = similarities[0][0]
            return nearest
        else:
            return ""
        

    def predict_improve(self, context): 
        stop_words = stopwords.words('english')
        # Based on context, get lemma, POS, left & right context
        lemma = context.lemma
        pos = context.pos
        left_context = context.left_context
        right_context = context.right_context
        
        new_context = left_context[-4:] + right_context[:4]  # Narrower the window size to +-4 words
        new_context = [word.lower() for word in new_context] # Convert into lowercase
        new_context = [word for word in new_context if word not in stop_words]  # Remove stop words
        new_context = [word for word in new_context if word not in string.punctuation]  # Remove punctuationS
        
        # sentence vector
        sen_vec = self.model.wv[lemma].copy()
        for word in new_context:
            if word in self.model.vocab:
                sen_vec += self.model.wv[word]

        # Obtain all possible synonyms
        possible_synonyms = get_candidates(lemma, pos)
        
        similarities = []
        # Iterate over possible synonyms，calculate similarities
        for p_synonym in possible_synonyms:
            p_synonym_restore = p_synonym.replace(' ', '_')
            # Check if the lemme belongs to the model
            if p_synonym_restore in self.model.vocab:
                similarity = self.cos(self.model.wv[p_synonym_restore], sen_vec)
            else:
                similarity = 0
            similarities.append((p_synonym, similarity))
        # Order by similarities
        similarities.sort(key=lambda x: x[1], reverse=True)
        # If there exists possible synonyms，return the one with highest similarity
        if similarities:
            nearest = similarities[0][0]
            return nearest
        else:
            return ""


if __name__=="__main__":

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        # prediction = smurf_predictor(context) 
        # prediction = wn_frequency_predictor(context)
        # prediction = wn_simple_lesk_predictor(context)
        # prediction = predictor.predict_nearest(context)
        # prediction = predictor.predict_nearest_with_context(context)
        prediction = predictor.predict_improve(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
