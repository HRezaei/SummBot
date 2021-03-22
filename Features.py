"""
Container for all sentence metrics
"""
import math
#from hazm import *



def pos_ratio_based_en(features, sentence_words):
    import nltk
    tags = nltk.pos_tag(sentence_words, tagset='universal')
    all_count = len(sentence_words)
    nn_count = sum(1 if tag=='NOUN' else 0 for (w, tag) in tags)
    ve_count = sum(1 if tag=='VERB' else 0 for (w, tag) in tags)
    aj_count = sum(1 if tag=='ADJ' or tag=='AJe' else 0 for (w, tag) in tags)
    av_count = sum(1 if tag=='ADV' else 0 for (w, tag) in tags)
    num_count = sum(1 if tag == 'NUM' else 0 for (w, tag) in tags)
    features['num_count'] = num_count
    features['pos_nn_ratio'] = nn_count/all_count
    features['pos_ve_ratio'] = ve_count/all_count
    features['pos_aj_ratio'] = aj_count/all_count
    features['pos_av_ratio'] = av_count/all_count
    features['pos_num_ratio'] = num_count/all_count
    features['nnf_isnnf'] = (nn_count/features['doc_nouns']) if features['doc_nouns'] > 0 else 0
    features['vef_isvef'] = (ve_count/features['doc_verbs']) if features['doc_verbs'] > 0 else 0
    features['ajf_isajf'] = (aj_count/features['doc_adjcs']) if features['doc_adjcs'] > 0 else 0
    features['avf_isavf'] = (av_count/features['doc_advbs']) if features['doc_advbs'] > 0 else 0
    features['nuf_isnuf'] = (num_count / features['doc_nums']) if features['doc_nums'] > 0 else 0


def frequency_score(sentence_words, word_freq):
    """
    Term Frequency measure, average
    Args:
        sentence: An array of tokenized words of the sentence
    """
    sen_score = 0
    for sen_word in sentence_words:
        sen_score = sen_score + word_freq[sen_word]
    return sen_score / len(sentence_words)


def inverse_sentence_freq(term, sentences):
    """
    Computes ISF
    Args:
        term: the word for which isf will be calculated
        sentences: array of all sentences in the text, tokenized and removed stop words
    """
    sentences_containing = 0
    for sen in sentences:
        if term in sen:
            sentences_containing = sentences_containing + 1
    if sentences_containing == 0:
        sentences_containing = 1
    return math.log(len(sentences)/sentences_containing)


def inverse_sentence_freq_old(term, sentences):
    """
    Computes ISF
    Until 06/26/2020, this variation of isf was used in our code, however I couldn't remember why we had devised such a
    formula and I couldn't find it in any reference! so I suffixed it with _old and put here for later reference.
    A more standard formula is used now in inverse_sentence_freq()
    Args:
        term: the word for which isf will be calculated
        sentences: array of all sentences in the text, tokenized and removed stop words
    """
    sentences_containing = 0
    for sen in sentences:
        if term in sen:
            sentences_containing = sentences_containing + 1
    if sentences_containing == 0:
        sentences_containing = 1
    return 1 - (math.log(sentences_containing) / math.log(len(sentences)))


def tf_isf_score(sentence_words, sentences, word_freq):
    sen_score = 0
    for sen_word in sentence_words:
        sen_score = sen_score + word_freq[
            sen_word] * inverse_sentence_freq(sen_word, sentences)
    return sen_score/len(sentence_words)


def linear_poition_score(position, total_sentences):
    return 1 - (position / total_sentences)


def cosine_position_score(position, total_sentences):
    alpha = 2
    return (math.cos(
        (2 * 3.14 * position) / (total_sentences - 1)) + alpha - 1) / alpha


def title_similarity_score(sen, title):
    denominator = math.sqrt(len(sen) * len(title))
    if denominator > 0:
        ratio = len(set(sen).intersection(title)) / denominator
    else:
        ratio = 0
    return ratio


def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity of the given parameters
    Args:
        vec1: frequency distribution of a sentence
        vec2: frequency distribution of a sentence
    """
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def cue_words(sentence_words, cue_words_list):
    '''

    '''
    output = 0
    for word in sentence_words:
        if word in cue_words_list:
            output += 1
    return output
