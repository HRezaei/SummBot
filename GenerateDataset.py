from nltk.probability import FreqDist
import json, sys, hashlib
from rouge import Rouge
from utilities import *
#from FarsnetLoader import *
import Features
import numpy as np
import hazm
import doc_per as farsi






from Features import *


cue_words = read_file("resources/cue-words.txt").split()
rouge = Rouge()



#farsnet = importEFromPaj("resources/farsnet/synset_related_to.paj")


#normalizer = Normalizer()
#stemmer = Stemmer()


def generate_dataset():    
    feats = build_feature_set()
    f_file = open('features.json', '+w')
    json.dump(feats, f_file, ensure_ascii=False, default=encode_complex)
    f_file.close()
    print("features.json has been written successfully")
    '''f_file = open('referense_sens.json', '+w')
    json.dump(refs, f_file, ensure_ascii=False, default=encode_complex)
    f_file.close()'''
    write_dataset_csv(feats, 'dataset.csv')


#if len(sys.argv) > 1 and sys.argv[1] == 'all':
#    generate_dataset()
