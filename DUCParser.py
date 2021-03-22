from utilities import read_file
import xml.etree.ElementTree as ET
import os, json

d = 0
l = 0
duc = {}
for file in os.listdir("resources/DUC/2007 SCU-marked corpus/"):
    if file.endswith(".scu"):
        path = os.path.join("resources/DUC/2007 SCU-marked corpus/", file)
        #content = read_file(path)
        root = ET.parse(path).getroot()
        for doc in root.findall('document'):
            document = {'text': [], 'summaries': {}}
            summaries = {}
            d += 1
            l=-1
            for line in doc.findall('line'):
                l += 1
                i = 0
                for annotation in line.findall('annotation'):
                    sums = annotation.get('sums').split(',')
                    for s in sums:
                        if s in summaries:
                            summaries[s].append(l)
                        else:
                            summaries[s] = [l]
                document['text'].append(line.text)
            document['summaries'] = summaries
            key = doc.get('name')
            duc[key] = document

'''fo = open('resources/DUC/all.json', 'w+')
json.dump(duc, fo)'''

from DUCGenerateFeatures import build_feature_set
build_feature_set()
