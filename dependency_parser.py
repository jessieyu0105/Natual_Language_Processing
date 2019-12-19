from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # Dictionary from indices to output actions 
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)
        # print(state)
        while state.buffer:
            features = self.extractor.get_input_representation(words, pos, state)
            possible_actions = self.model.predict(np.vstack([features]))
            probs = [(idx, prob) for idx, prob in enumerate(possible_actions[0])]
            # Sort it according to the output probabilities
            sorted_probs = sorted(probs, key=lambda x: x[1], reverse=True)
            # print(sorted_probs)
            for sp in sorted_probs:
                output_label = self.output_labels[sp[0]]
                transition, label = output_label
                if transition == 'shift':
                    # Check illegal case 1: Shifting the only word out of the buffer is illegal, unless the stack is empty
                    if len(state.buffer) == 1 and state.stack != []:
                        continue
                    state.shift()
                    break
                # Check illegal case 2: arc-left is not permitted if the stack is empty
                # Check illegal case 3: the root node must never be the target of a left-arc
                elif transition == 'left_arc' and state.stack != [] and state.stack[-1] != 0:
                    state.left_arc(label)
                    break
                # Check illegal case 2: arc-right is not permitted if the stack is empty
                elif transition == 'right_arc' and state.stack != []:
                    state.right_arc(label)
                    break

        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file:  # sys.argv[2]: dev.conll  the second parameter while running the .py
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
