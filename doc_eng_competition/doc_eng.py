import xml.etree.ElementTree as ET
import os, json
from utilities import *
from html import unescape

'''
@todo: Move category mapping to the phase in which csv is written
@todo: review this problem: When in a run, I call parse_dataset(), extract_features(), and learn() I get an exception in summ() line below:
    row.append(sen[attr])
    saying that tuple indices must be integers not str! 
'''


def read_cnn_directory(corpus_path):

    #corpus_path = '/home/hrezaei/Documents/CNN_Corpus'

    files = os.listdir(corpus_path)
    cats = set()
    corpus = {}
    num_all_sentences = 0
    num_summary_sentences = 0
    for file in files:
        tree = ET.parse(corpus_path + '/' + file)
        root = tree.getroot()
        doc_id = root.attrib['id']
        category = root.attrib['category']
        num_paragraphs = len(root.findall('./article/paragraph'))
        sentences = []
        for item in root.findall('./article/paragraph/sentences/sentence/content'):
            sentences.append(unescape(item.text.replace('&apost;', '&apos;')))
        summary = []
        for item in root.findall('./summaries/gold_standard/sentence'):
            summary.append(unescape(item.text.replace('&apost;', '&apos;')))
        highlights = []
        for item in root.findall('./summaries/highlights/sentence'):
            highlights.append(unescape(item.text.replace('&apost;', '&apos;')))
        corpus[int(doc_id)] = {
            'file_name': file,
            'title': unescape(root.find('title').text.replace('&apost;', '&apos;')),
            'keywords': unescape(root.find('keywords').text.replace('&apost;', '&apos;')) if root.find('keywords').text else '',
            'sentences': sentences,
            'summaries': {'gold': summary},  # , 'highlights': highlights'''
            'category': category,
            'num_paragraphs': num_paragraphs
        }
        num_all_sentences += len(sentences)
        num_summary_sentences += len(summary)
        #if len(corpus) > 9:
        #   break
    #print(cats)
    #fp = open('cnn.json', 'w')
    #json.dump(corpus, fp)
    print('Number of sentences in summaries', num_summary_sentences)
    print('Number of all sentences', num_all_sentences)
    return corpus


def build_feature_set():
    cnn_dataset = json_read('resources/CNN/documents.json')
    output = {}
    num_all_feature_vectors = 0
    num_true_vectors = 0
    for doc_id in sorted(cnn_dataset):
        doc = cnn_dataset[doc_id]
        feature_set, tmp = document_feature_set(doc, doc['summaries'], doc_id)
        num_all_feature_vectors += len(feature_set)
        num_true_vectors += sum([1 if label else 0 for (v, label) in feature_set])
        output[doc_id] = feature_set

    print('Number of all feature vectors', num_all_feature_vectors)
    print('Number of vectors labeled True:', num_true_vectors)
    return output


def document_feature_set(doc, golden_summaries=[], key=''):
    """
    Converts a raw text to a matrix of features.
    Each row corresponds with a sentence in given text
    If golden summaries is passed, it also computes target attributes and a few other additional features
    This function is used both in generating dataset and in summarizing an individual text
    :param text:
    :param category:
    :param golden_summaries:
    :param key:
    :return:
    """
    import hashlib

    text_sentences = doc['sentences']
    category = doc['category']
    num_paragraphs = doc['num_paragraphs']
    title = full_preprocess(doc['title'])
    keywords = full_preprocess(doc['keywords'])
    stemmer = english_stemmer()
    title_stems = [stemmer.stem(w) for w in title]
    keyword_stems = [stemmer.stem(w) for w in keywords]

    text = ' '.join(text_sentences)
    hash_key = hashlib.md5((text+category).encode('utf-8')).hexdigest()
    if hash_key in document_feature_set.cache:
        return document_feature_set.cache[hash_key]
    feature_set = []
    sentences_words = []
    sentences_words_stems = []
    tagged_sentences = []
    num_verbs = 0  # in doc
    num_nouns = 0
    num_advbs = 0
    num_adjcs = 0
    doc_nums = 0
    all_words = full_preprocess(text)
    #tags = nltk.pos_tag(words, tagset='universal')
    all_words_stemmed = [stemmer.stem(w) for w in all_words]
    word_freq = nltk.FreqDist(all_words_stemmed)

    for sen in text_sentences[:]:
        words = full_preprocess(sen)
        if len(words) < 1:
            text_sentences.remove(sen)
            continue
        sentences_words.append(words)
        sentences_words_stems.append([stemmer.stem(w) for w in words])
        tagged_sen = nltk.pos_tag(words, tagset='universal')
        num_nouns += sum(1 if tag == 'NOUN' else 0 for (w, tag) in tagged_sen)
        num_verbs += sum(1 if tag == 'VERB' else 0 for (w, tag) in tagged_sen)
        num_adjcs += sum(1 if tag == 'ADJ' or tag == 'AJe' else 0 for (w, tag) in tagged_sen)
        num_advbs += sum(1 if tag == 'ADV' else 0 for (w, tag) in tagged_sen)
        doc_nums += sum(1 if tag == 'NUM' else 0 for (w, tag) in tagged_sen)
        tagged_sentences.append(tagged_sen)

    doc_features = {
        'doc_words': len(all_words),
        'doc_sens': len(sentences_words),
        'doc_parag': num_paragraphs,
        'category': category,
        'doc_verbs': num_verbs,
        'doc_adjcs': num_adjcs,
        'doc_advbs': num_advbs,
        'doc_nouns': num_nouns,
        'doc_nums': doc_nums,
        'category_us': category == 'us',
        'category_world': category == 'world',
        'category_showbiz': category == 'showbiz',
        'category_living': category == 'living',
        'category_travel': category == 'travel',
        'category_business': category == 'business',
        'category_justice': category == 'justice',
        'category_health': category == 'health',
        'category_sport': category == 'sport',
        'category_politics': category == 'politics',
        'category_tech': category == 'tech',
        'category_opinion': category == 'opinion'
    }

    if golden_summaries:
        normalized_summaries = []
        gold_summaries = {}
        for summ in golden_summaries:
            summ_sens = golden_summaries[summ]
            # word tokenized summaries for computing bleu scores:
            normalized_summaries.append(' '.join(summ_sens).split())
            words_processed = [full_preprocess(sen) for sen in summ_sens]
            words_stemmed = [[stemmer.stem(w) for w in ref_sen] for ref_sen in words_processed]
            gold_summaries[summ] = {
                'sens': words_processed,
                'sens_stemmed': words_stemmed
            }

    position = 0
    for words in sentences_words:
        sen_stems = sentences_words_stems[position]
        document_feature_set.id += 1
        features = doc_features.copy()
        add_features(features, sentences_words, sentences_words_stems, word_freq, position)
        features['id'] = document_feature_set.id
        features['id_ref'] = str(key) + '_' + str(position)
        features['title_resemblance'] = are_similar(sen_stems, title_stems)[1]
        features['keywords_resemblance'] = are_similar(sen_stems, keyword_stems)[1]
        if golden_summaries:
            features['target_bleu'] = str(avg_bleu_score(words, normalized_summaries))
            features['target_bleu_avg'] = avg_bleu_score(words, normalized_summaries, True)
            gold_sentences_stemmed = [g['sens_stemmed'] for g in gold_summaries]
            features['target'] = average_similarity(sen_stems, gold_sentences_stemmed)
            included = (features['target'] > similarity_threshold)
            features['included'] = included
            features['source_file'] = key
            features['text'] = ' '.join(words)
            feature_set.append((features, included))
        else:
            feature_set.append(features)
        position += 1

    document_feature_set.cache[hash_key] = (feature_set, text_sentences)
    return feature_set, text_sentences


document_feature_set.id = 0
document_feature_set.cache = {}


def full_preprocess(sentence, remove_stopwords=True):
    all_words = nltk.wordpunct_tokenize(sentence)
    all_words = [w.lower() for w in all_words if w.isalnum()]
    if remove_stopwords:
        if len(full_preprocess.stopwords_cache) == 0:
            full_preprocess.stopwords_cache = set(nltk.corpus.stopwords.words('english'))

        all_words = [w for w in all_words if w not in full_preprocess.stopwords_cache]
    return all_words


full_preprocess.stopwords_cache = []


def remove_stop_words_and_punc(all_words):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    if isinstance(all_words, str):
        all_words = nltk.wordpunct_tokenize(all_words)

    return [w for w in all_words if w not in stop_words and w.isalnum()]


def add_features(features, all_sentences_tokenized, all_sentences_stemmed, word_freq, position):
    '''
    Args:
        sent: array of words
    '''
    import Features
    sent_stems = all_sentences_stemmed[position]
    sent_words = all_sentences_tokenized[position]
    total_sentences = len(all_sentences_tokenized)
    features["tfisf"] = Features.tf_isf_score(sent_stems, all_sentences_stemmed, word_freq)
    features["cosine_position"] = Features.cosine_position_score(position, total_sentences)
    features['position'] = 1/(position+1)
    features["tf"] = Features.frequency_score(sent_stems, word_freq)
    features["cue_words"] = Features.cue_words(sent_words, cue_words('en'))
    features['len'] = len(sent_words)
    avg_len = sum([len(s) for s in all_sentences_tokenized])/total_sentences
    features['relative_len'] = len(sent_words)/avg_len
    Features.pos_ratio_based_en(features, sent_words)
    return features



def write_xml(summary, file_id, file_name, directory):
    import xml.etree.ElementTree as etree

    top = etree.Element('document')
    top.attrib['id'] = str(file_id)
    comment = etree.Comment('Generated for DocEng 2020 competition')
    top.append(comment)

    summaries = etree.SubElement(top, 'summaries')

    for (i, sen) in summary:
        child_sen = etree.SubElement(summaries, 'sentence')
        child_sen.attrib['id'] = str(i)
        child_sen.text = sen

    fp = open(directory + file_name, 'wb')
    e = etree.ElementTree(top)
    e.write(fp)
    fp.close()





