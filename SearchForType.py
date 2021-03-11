# coding=utf-8

import xml.etree.cElementTree as ETree
import os
import MySQLdb
import nltk
from nltk.corpus import stopwords
import re
from scipy.linalg import norm
import numpy as np
import gensim
#from spacy.lang.en import English
import spacy
import re
from xml.dom.minidom import Document
from xml.dom import minidom
from bert_serving.client import BertClient
import pandas as pd
from sklearn import model_selection,svm,metrics
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
stop_words = stopwords.words('english')
for w in ['!',',','.','?','-s','-ly','</s>','s']:
    stop_words.append(w)
#model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
#np.seterr(divide='ignore',invalid='ignore')

class strandform:

    def __init__(self):
        self.types = None
        self.verb = None
        self.step = None
        self.clickwhere = list()
        self.clicktype = None
        self.clicktimes = False
        self.typewhat = list()
        self.digittypewhat = list()
        self.typewhere = list()
        self.digittypewhere = list()
        self.typetimes = False
        self.createwhat = list()
        self.sentence = list()
        self.sent = None
        self.sentenceid = None

    def printform(self):
        if (self.types == 0):
            print("click type")
        if (self.types == 1):
            print("click click")
        if (self.types == 2):
            print("click create")
        return 2


def pickupcrital(juzi):
    SUBJECTS = ["nsubj", "nsubjpass"]  ## add or delete more as you wish
    OBJECTS = ["dobj", "pobj", "dobj"]  ## add or delete more as you wish
    PREDICATE = ["VERB"]

    #nlp = English()
    nlp = spacy.load('en')
    sents = []
    ############all cases
    # file_allcase = open(address+"/middleResults/allcases.xml","wb")
    #file_allcase = open(address + "/middleResults/allcases.xml", 'w')
    doc_allcase = Document()
    root_allcase = doc_allcase.createElement("Allcases")

    ##########################

    recorder = None  # record last sentence
    steplist = list()
    step = 0
    sentid = 0

    for sent in nlp(juzi.lower()).sents:
        print(sent)
        ########all cases
        newsent = doc_allcase.createElement("Sent" + str(sentid))
        root_allcase.appendChild(newsent)

        sentenceid = doc_allcase.createElement("Sentid")
        nodetext = doc_allcase.createTextNode(str(sentid))
        sentenceid.appendChild(nodetext)
        newsent.appendChild(sentenceid)

        sentid += 1

        sentele = doc_allcase.createElement("sentence")
        newsent.appendChild(sentele)

        nounlist = doc_allcase.createElement("nounlist")
        newsent.appendChild(nounlist)

        #########

        stringnounphrasemap = {}
        normalnounphrasemap = {}

        # doc=nlp(unicode(sent))
        doc = nlp(str(sent))
        recordstring = list()
        recordindexlist = list()
        stringboolean = False
        for word in doc:
            if (word.text == "\"" and stringboolean == False):
                stringboolean = True
                continue
            if (word.text == "\"" and stringboolean == True):
                stringboolean = False
                for index in recordindexlist:
                    stringnounphrasemap[index] = recordstring
                recordstring = list()  # initial
                recordindexlist = list()
                continue

            if (stringboolean == True):
                recordstring.append(word.text)
                recordindexlist.append(word.i)

        for np in doc.noun_chunks:
            recordstring = list()
            recordindexlist = list()
            i = 0
            for word in np:
                if (word.pos_ == u"PUNCT"):
                    continue
                i += 1
            if (i > 1):
                for word in np:
                    if (not word.pos_ == u"PUNCT"):
                        recordstring.append(word.lemma_)
                        recordindexlist.append(word.i)
                for index in recordindexlist:
                    normalnounphrasemap[index] = recordstring

        # test codes
        # for k,v in normalnounphrasemap.items():
        #    print(k,v)

        newinfo = None

        nounk = 0
        kk = 0
        for word in doc:

            ###########above all case##############################
            newitem = doc_allcase.createElement("item" + str(kk))
            noteText = doc_allcase.createTextNode(word.lemma_)
            newitem.appendChild(noteText)
            sentele.appendChild(newitem)

            if word.pos_ in ["NOUN"]:
                nounitem = doc_allcase.createElement("item" + str(nounk))
                nounk += 1
                noteText = doc_allcase.createTextNode(word.lemma_)
                nounitem.appendChild(noteText)
                nounlist.appendChild(nounitem)

                if word.head.pos_ in ["VERB"]:
                    nounitem.setAttribute("verb", word.head.lemma_)

            ############example case#########################
            if not recorder == None:
                if recorder.types == "input":

                    examplecase = False
                    for eachword in doc:
                        if (eachword.lemma_ in ["e.g.", "example", "say"]):
                            examplecase = True
                            break

                    if examplecase == True:
                        if (word.pos_ in ["NUM"] and word.is_digit == False):
                            # print("test1")
                            # print(word.text)
                            recorder.digittypewhat.append(re.findall('[\d ]+', word.lemma_))  # digit
                            recorder.digittypewhere.append(re.findall('[^\d ]+', word.lemma_))  # string

                            # print re.findall ('[\d ]+', word.text);
                            # print re.findall ('[^\d ]+', word.text);

                        if (word.pos_ in ["NUM"] and word.is_digit == True):
                            recorder.digittypewhat.insert(0, word.lemma_)
                            # recorder.digittypewhere.append(word.head.text)
                            # for child in word.children:
                            recorder.digittypewhere.insert(0, doc.__getitem__(word.i + 1))  # get next item

                            # print("test2")
                            # print(word.text)
                            # print word.head.text
            # print(type(word.lemma_))
            # print(type(sent))
            #result1 = str(word.lemma_) + ' _ ' + str(sent)
            #types = getClass(result1)
            #########################click case####################
            if(word.lemma_ in ["click","choose","select","lanuch","pick","tap","open","press","go"]):
                #print(word.lemma_ )
                newinfo = strandform()
                newinfo.types = "click"
                newinfo.step = step
                newinfo.verb = str(word.lemma_ )
                newinfo.sent = sent

                for eachword in doc:
                    newinfo.sentence.append(eachword.lemma_)

                newinfo.sentenceid = sentid - 1

                # newinfo.sentence=sent

                step += 1

                ####8/8/2017 fix the if part
                if word.dep_ in ["compound"]:
                    # if stringnounphrasemap.has_key(word.head.i):
                    if word.head.i in stringnounphrasemap:
                        newinfo.clickwhere.append(stringnounphrasemap.get(word.head.i))
                    # if normalnounphrasemap.has_key(word.head.i):
                    if word.head.i in normalnounphrasemap:
                        newinfo.clickwhere.append(normalnounphrasemap.get(word.head.i))
                    newinfo.clickwhere.append(word.head.lemma_)

                elif word.dep_ in ["amod"]:
                    # if stringnounphrasemap.has_key(word.head.i):
                    if word.head.i in stringnounphrasemap:
                        newinfo.clickwhere.append(stringnounphrasemap.get(word.head.i))
                    # if normalnounphrasemap.has_key(word.head.i):
                    if word.head.i in normalnounphrasemap:
                        newinfo.clickwhere.append(normalnounphrasemap.get(word.head.i))
                    newinfo.clickwhere.append(word.head.lemma_)

                else:
                    for child in word.subtree:
                        if (child.dep_ in ["nsubjpass", "dobj", "pobj", "oprd",
                                           "intj"]):  # injt gantanci  #oprd is a ind of object
                            # if stringnounphrasemap.has_key(child.i):# check weather the hashmap contain or not
                            if child.i in stringnounphrasemap:
                                newinfo.clickwhere.append(stringnounphrasemap.get(child.i))

                            # if normalnounphrasemap.has_key(child.i):
                            if child.i in normalnounphrasemap:
                                newinfo.clickwhere.append(normalnounphrasemap.get(child.i))

                            newinfo.clickwhere.append(child.lemma_)

                for subtr in word.subtree:
                    if (subtr.lemma_ in [u"multiple", u"twice"] or subtr.text in [u"times"]):
                        newinfo.clicktimes = True

                    if (subtr.lemma_ in ["long"]):
                        newinfo.clicktype = "long"

                steplist.append(newinfo)
            ######################################################
            if (word.lemma_ in ["cancel"]):
                newinfo = strandform()
                newinfo.types = "cancel"
                newinfo.verb = str(word.lemma_ )
                newinfo.step = step
                newinfo.sent = sent
                for eachword in doc:
                    newinfo.sentence.append(eachword.lemma_)

                newinfo.sentenceid = sentid - 1

                step += 1
                steplist.append(newinfo)

            #######################create case#####################
            if (word.lemma_ in ["create"]):
                newinfo = strandform()
                newinfo.types = "create"
                newinfo.verb = str(word.lemma_ )
                newinfo.step = step
                newinfo.sent = sent
                for eachword in doc:
                    newinfo.sentence.append(eachword.lemma_)

                newinfo.sentenceid = sentid - 1

                step += 1

                ####8/8/2017 fix the if part
                if word.dep_ in ["compound"]:
                    # if stringnounphrasemap.has_key(word.head.i):
                    if word.head.i in stringnounphrasemap:
                        newinfo.createwhat.append(stringnounphrasemap.get(word.head.i))
                    # if normalnounphrasemap.has_key(word.head.i):
                    if word.head.i in normalnounphrasemap:
                        newinfo.createwhat.append(normalnounphrasemap.get(word.head.i))
                    newinfo.createwhat.append(word.head.lemma_)

                elif word.dep_ in ["amod"]:
                    # if stringnounphrasemap.has_key(word.head.i):
                    if word.head.i in stringnounphrasemap:
                        newinfo.createwhat.append(stringnounphrasemap.get(word.head.i))
                    # if normalnounphrasemap.has_key(word.head.i):
                    if word.head.i in normalnounphrasemap:
                        newinfo.createwhat.append(normalnounphrasemap.get(word.head.i))
                    newinfo.createwhat.append(word.head.lemma_)

                else:

                    for child in word.children:
                        if (child.dep_ in ["nsubjpass", "dobj"]):
                            # if stringnounphrasemap.has_key(child.i):# check weather the hashmap contain or not
                            if child.i in stringnounphrasemap:
                                newinfo.createwhat.append(stringnounphrasemap.get(child.i))

                            # if normalnounphrasemap.has_key(child.i):
                            if child.i in normalnounphrasemap:
                                newinfo.createwhat.append(normalnounphrasemap.get(child.i))

                            newinfo.createwhat.append(child.lemma_)
                            # subtreelist=list()
                            for sub in child.children:
                                if (sub.dep_ in ["prep"]):
                                    for subsub in sub.children:
                                        if (subsub.dep_ in ["pobj"]):
                                            newinfo.createwhat.append(subsub.lemma_)

                steplist.append(newinfo)
            #####################rotate###########################
            if(word.lemma_ in ["rotate", "orientation"]):
                newinfo = strandform()
                newinfo.types = "rotate"
                newinfo.sent = sent
                newinfo.verb = str(word.lemma_ )
                newinfo.step = step
                for eachword in doc:
                    newinfo.sentence.append(eachword.lemma_)

                newinfo.sentenceid = sentid - 1

                step += 1
                steplist.append(newinfo)

            ###########################left out case#############################
            if (word.lemma_ in ["leave", "left"]):
                if (doc.__getitem__(kk + 1).lemma_ in ["out"]):

                    newinfo = strandform()
                    newinfo.types = "input"
                    newinfo.verb = str(word.lemma_ )
                    newinfo.step = step
                    newinfo.sentenceid = sentid - 1
                    for everyword in doc:
                        newinfo.sentence.append(everyword.lemma_)

                    step += 1

                    newinfo.typewhat.append("")
                    # newinfo.typewhere.append(stringnounphrasemap.get(child.i))

                    for wordchild in word.subtree:
                        # if (wordchild.dep_ in ["dobj"]):
                        newinfo.typewhere.append(wordchild.lemma_)

                    newinfo.digittypewhat.append("")
                    # newinfo.typewhere.append(stringnounphrasemap.get(child.i))

                    for wordchild in word.subtree:
                        # if (wordchild.dep_ in ["dobj"]):
                        newinfo.digittypewhere.append(wordchild.lemma_)
                        # newinfo.typewhere.append(stringnounphrasemap.get(wordchild))

                    #    for (child.dep_ in [ ]):

                    steplist.append(newinfo)

            ###########################input case###################3
            if(word.lemma_ in ["input","enter","type","insert","fill","change","write","set","put","add"]):
                # print("word is an input")
                newinfo = strandform()
                newinfo.types = "input"
                newinfo.verb = str(word.lemma_ )
                newinfo.step = step
                newinfo.sent = sent

                for everyword in doc:
                    newinfo.sentence.append(everyword.lemma_)
                    # newinfo.sentence.append(stringnounphrasemap.get(everyword.lemma_))

                newinfo.sentenceid = sentid - 1

                step += 1
                for subtr in word.subtree:
                    if (subtr.lemma_ in [u"multiple", u"twice"] or subtr.text in [u"times"]):
                        newinfo.typetimes = True
                        #######################"input is NOUN"
                if (word.pos_ in [u"NOUN"]):
                    rootofdoc = None
                    for oneword in doc:  # find the root
                        if (oneword.dep_ == u'ROOT'):
                            rootofdoc = oneword

                    for child in doc:
                        if (child.dep_ in [u"dobj", u"obj", u"attr", u"appos", u"nmod"]):

                            for prepchild in doc:
                                if (prepchild.dep_ in [u"prep"] and prepchild.lemma_ in [u"on", u"in", u"to"]):

                                    # if stringnounphrasemap.has_key(child.i):# check weather the hashmap contain or not
                                    if child.i in stringnounphrasemap:
                                        newinfo.typewhat.append(stringnounphrasemap.get(child.i))

                                    # if normalnounphrasemap.has_key(child.i):
                                    if child.i in normalnounphrasemap:
                                        newinfo.typewhat.append(normalnounphrasemap.get(child.i))

                                    # 28/7/2017 added
                                    sublist = list()
                                    for sub in child.subtree:
                                        if sub.pos_ in [u"NUM"]:
                                            sublist.append(sub.lemma_)
                                    if (len(sublist) > 0):
                                        newinfo.typewhat.append(sublist)
                                    # end

                                    newinfo.typewhat.append(child.lemma_)

                                    for onobjchild in prepchild.children:
                                        if (onobjchild.dep_ in [u"pobj"]):
                                            # if stringnounphrasemap.has_key(onobjchild.i):
                                            if onobjchild.i in stringnounphrasemap:
                                                newinfo.typewhere.append(stringnounphrasemap.get(onobjchild.i))

                                            if onobjchild.i in normalnounphrasemap:
                                                newinfo.typewhere.append(normalnounphrasemap.get(onobjchild.i))

                                            # 28/7/2017 added
                                            sublist = list()
                                            for sub in onobjchild.subtree:
                                                if sub.pos_ in [u"NUM"]:
                                                    sublist.append(sub.lemma_)
                                            if (len(sublist) > 0):
                                                newinfo.typewhere.append(sublist)
                                            # end

                                            newinfo.typewhere.append(onobjchild.lemma_)

                                elif (prepchild.dep_ in [u"prep"] and prepchild.lemma_ in [u"with"]):
                                    # if stringnounphrasemap.has_key(child.i):# check weather the hashmap contain or not
                                    if child.i in stringnounphrasemap:
                                        newinfo.typewhere.append(stringnounphrasemap.get(child.i))

                                    # if normalnounphrasemap.has_key(child.i):
                                    if child.i in normalnounphrasemap:
                                        newinfo.typewhere.append(normalnounphrasemap.get(child.i))

                                    # 28/7/2017 added
                                    sublist = list()
                                    for sub in child.subtree:
                                        if sub.pos_ in [u"NUM"]:
                                            sublist.append(sub.lemma_)
                                    if (len(sublist) > 0):
                                        newinfo.typewhere.append(sublist)
                                    # end

                                    newinfo.typewhere.append(child.lemma_)

                                    for onobjchild in prepchild.children:
                                        if (onobjchild.dep_ in [u"pobj"]):
                                            # if stringnounphrasemap.has_key(onobjchild.i):
                                            if onobjchild.i in stringnounphrasemap:
                                                newinfo.typewhat.append(stringnounphrasemap.get(onobjchild.i))

                                            # if normalnounphrasemap.has_key(onobjchild.i):
                                            if onobjchild.i in normalnounphrasemap:
                                                newinfo.typewhat.append(normalnounphrasemap.get(onobjchild.i))

                                            # 28/7/2017 added
                                            sublist = list()
                                            for sub in onobjchild.subtree:
                                                if sub.pos_ in [u"NUM"]:
                                                    sublist.append(sub.lemma_)
                                            if (len(sublist) > 0):
                                                newinfo.typewhat.append(sublist)
                                            # end

                                            newinfo.typewhat.append(onobjchild.lemma_)

                #######################the input is not a NOUN#############
                for child in word.children:
                    #########################passive
                    if (child.dep_ in [u"nsubjpass"]):  # passive case
                        # for prepchild in word.children:# notice that the word.children
                        for prepchild in doc:
                            if (prepchild.dep_ in [u"prep"] and prepchild.lemma_ in [u"on", u"in", u"to"]):

                                # if stringnounphrasemap.has_key(child.i):# check weather the hashmap contain or not
                                if child.i in stringnounphrasemap:
                                    if word.lemma_ in ["change"]:  # change has a inverse grammar than fill
                                        newinfo.typewhere.append(stringnounphrasemap.get(child.i));
                                    else:
                                        newinfo.typewhat.append(stringnounphrasemap.get(child.i))

                                # if normalnounphrasemap.has_key(child.i):
                                if child.i in normalnounphrasemap:
                                    if word.lemma_ in ["change"]:
                                        newinfo.typewhere.append(normalnounphrasemap.get(child.i))
                                    else:
                                        newinfo.typewhat.append(normalnounphrasemap.get(child.i))

                                # 28/7/2017 added
                                sublist = list()
                                for sub in child.subtree:
                                    if sub.pos_ in [u"NUM"]:
                                        sublist.append(sub.lemma_)
                                if (len(sublist) > 0):
                                    newinfo.typewhat.append(sublist)
                                # end

                                newinfo.typewhat.append(child.lemma_)

                                for onobjchild in prepchild.children:
                                    if (onobjchild.dep_ in [u"pobj"]):
                                        # if stringnounphrasemap.has_key(onobjchild.i):
                                        if onobjchild.i in stringnounphrasemap:
                                            if word.lemma_ in ["change"]:
                                                newinfo.typewhat.append(stringnounphrasemap.get(onobjchild.i))
                                            else:
                                                newinfo.typewhere.append(stringnounphrasemap.get(onobjchild.i))

                                        # if normalnounphrasemap.has_key(onobjchild.i):
                                        if onobjchild.i in normalnounphrasemap:
                                            if word.lemma_ in ["change"]:
                                                newinfo.typewhat.append(normalnounphrasemap.get(onobjchild.i))
                                            else:
                                                newinfo.typewhere.append(normalnounphrasemap.get(onobjchild.i))

                                        # 28/7/2017 added
                                        sublist = list()
                                        for sub in onobjchild.subtree:
                                            if sub.pos_ in [u"NUM"]:
                                                sublist.append(sub.lemma_)
                                            if (len(sublist) > 0):
                                                newinfo.typewhere.append(sublist)
                                        # end

                                        newinfo.typewhere.append(onobjchild.lemma_)

                            elif (prepchild.dep_ in [u"prep"] and prepchild.lemma_ in [u"with"]):
                                # if stringnounphrasemap.has_key(child.i):# check weather the hashmap contain or not
                                if child.i in stringnounphrasemap:
                                    newinfo.typewhere.append(stringnounphrasemap.get(child.i))

                                # if normalnounphrasemap.has_key(child.i):
                                if child.i in normalnounphrasemap:
                                    newinfo.typewhere.append(normalnounphrasemap.get(child.i))

                                # 28/7/2017 added
                                sublist = list()
                                for sub in child.subtree:
                                    if sub.pos_ in [u"NUM"]:
                                        sublist.append(sub.lemma_)
                                    if (len(sublist) > 0):
                                        newinfo.typewhere.append(sublist)
                                # end

                                newinfo.typewhere.append(child.lemma_)

                                for onobjchild in prepchild.children:
                                    if (onobjchild.dep_ in [u"pobj"]):
                                        # if stringnounphrasemap.has_key(onobjchild.i):
                                        if onobjchild.i in stringnounphrasemap:
                                            newinfo.typewhat.append(stringnounphrasemap.get(onobjchild.i))

                                        # if normalnounphrasemap.has_key(onobjchild.i):
                                        if onobjchild.i in normalnounphrasemap:
                                            newinfo.typewhat.append(normalnounphrasemap.get(onobjchild.i))

                                        # 28/7/2017 added
                                        sublist = list()
                                        for sub in onobjchild.subtree:
                                            if sub.pos_ in [u"NUM"]:
                                                sublist.append(sub.lemma_)
                                            if (len(sublist) > 0):
                                                newinfo.typewhat.append(sublist)
                                        # end

                                        newinfo.typewhat.append(onobjchild.lemma_)
                    #################################active
                    if (child.dep_ in [u"pobj", u"dobj", u"appos"]):
                        for prepchild in doc:
                            if (prepchild.dep_ in [u"prep"] and prepchild.lemma_ in [u"on", u"in", u"to", u"as"]):

                                # if stringnounphrasemap.has_key(child.i):# check weather the hashmap contain or not
                                if child.i in stringnounphrasemap:
                                    if word.lemma_ in ["change"]:  # change has a inverse grammar than fill
                                        newinfo.typewhere.append(stringnounphrasemap.get(child.i))
                                    else:
                                        newinfo.typewhat.append(stringnounphrasemap.get(child.i))

                                # if normalnounphrasemap.has_key(child.i):
                                if child.i in normalnounphrasemap:
                                    if word.lemma_ in ["change"]:  # change has a inverse grammar than fill
                                        newinfo.typewhere.append(normalnounphrasemap.get(child.i))
                                    else:
                                        newinfo.typewhat.append(normalnounphrasemap.get(child.i))

                                # 28/7/2017 added
                                sublist = list()
                                for sub in child.subtree:
                                    if sub.pos_ in [u"NUM"]:
                                        sublist.append(sub.lemma_)
                                if (len(sublist) > 0):
                                    newinfo.typewhat.append(sublist)
                                # end

                                if word.lemma_ in ["change"]:  # change has a inverse grammar than fill
                                    newinfo.typewhere.append(child.lemma_)
                                else:
                                    newinfo.typewhat.append(child.lemma_)

                                for onobjchild in prepchild.children:
                                    if (onobjchild.dep_ in [u"pobj"]):
                                        # if stringnounphrasemap.has_key(onobjchild.i):
                                        if onobjchild.i in stringnounphrasemap:
                                            if word.lemma_ in ["change"]:  # change has a inverse grammar than fill
                                                newinfo.typewhat.append(stringnounphrasemap.get(onobjchild.i))
                                            else:
                                                newinfo.typewhere.append(stringnounphrasemap.get(onobjchild.i))

                                        # if normalnounphrasemap.has_key(onobjchild.i):
                                        if onobjchild.i in normalnounphrasemap:
                                            if word.lemma_ in ["change"]:  # change has a inverse grammar than fill
                                                newinfo.typewhat.append(normalnounphrasemap.get(onobjchild.i))
                                            else:
                                                newinfo.typewhere.append(normalnounphrasemap.get(onobjchild.i))

                                        # 28/7/2017 added
                                        sublist = list()
                                        for sub in onobjchild.subtree:
                                            if sub.pos_ in [u"NUM"]:
                                                sublist.append(sub.lemma_)
                                        if (len(sublist) > 0):
                                            newinfo.typewhere.append(sublist)
                                        # end

                                        if word.lemma_ in ["change"]:  # change has a inverse grammar than fill
                                            newinfo.typewhat.append(onobjchild.lemma_)
                                        else:
                                            newinfo.typewhere.append(onobjchild.lemma_)
                                        # newinfo.typewhere.append(onobjchild.lemma_)

                            elif (prepchild.dep_ in [u"prep"] and prepchild.lemma_ in [u"with"]):
                                # if stringnounphrasemap.has_key(child.i):# check weather the hashmap contain or not
                                if child.i in stringnounphrasemap:
                                    newinfo.typewhere.append(stringnounphrasemap.get(child.i))

                                # if normalnounphrasemap.has_key(child.i):
                                if child.i in normalnounphrasemap:
                                    newinfo.typewhere.append(normalnounphrasemap.get(child.i))

                                # 28/7/2017 added
                                sublist = list()
                                for sub in child.subtree:
                                    if sub.pos_ in [u"NUM"]:
                                        sublist.append(sub.lemma_)
                                    if (len(sublist) > 0):
                                        newinfo.typewhere.append(sublist)
                                # end

                                newinfo.typewhere.append(child.lemma_)

                                for onobjchild in prepchild.children:
                                    if (onobjchild.dep_ in [u"pobj"]):
                                        # if stringnounphrasemap.has_key(onobjchild.i):
                                        if onobjchild.i in stringnounphrasemap:
                                            newinfo.typewhat.insert(0, stringnounphrasemap.get(onobjchild.i));
                                            # newinfo.typewhat.append(stringnounphrasemap.get(onobjchild.i))

                                        # if normalnounphrasemap.has_key(onobjchild.i):
                                        if onobjchild.i in normalnounphrasemap:
                                            # newinfo.typewhat.append(normalnounphrasemap.get(onobjchild.i))
                                            newinfo.typewhat.insert(0, normalnounphrasemap.get(onobjchild.i));

                                        # 28/7/2017 added
                                        sublist = list()
                                        for sub in onobjchild.subtree:
                                            if sub.pos_ in [u"NUM"]:
                                                sublist.append(sub.lemma_)
                                            if (len(sublist) > 0):
                                                newinfo.typewhat.append(sublist)
                                        # end
                                        newinfo.typewhat.append(onobjchild.lemma_)
                            else:  ##2018.2.19
                                for onobjchild in prepchild.children:
                                    if (onobjchild.dep_ in [u"dobj", u"pobj"]):
                                        # if stringnounphrasemap.has_key(onobjchild.i):
                                        if onobjchild.i in stringnounphrasemap:
                                            if word.lemma_ not in [
                                                "change"]:  # change has a inverse grammar than fill
                                                newinfo.typewhat.append(stringnounphrasemap.get(onobjchild.i))

                                        # if normalnounphrasemap.has_key(onobjchild.i):
                                        if onobjchild.i in normalnounphrasemap:
                                            if word.lemma_ not in [
                                                "change"]:  # change has a inverse grammar than fill
                                                newinfo.typewhat.append(normalnounphrasemap.get(onobjchild.i))

                if not newinfo == None:
                    if newinfo.types == "input":
                        for oneword in doc:
                            ################digit case########################
                            if (oneword.pos_ in ["NUM"] and oneword.is_digit == False):
                                # print("test1")
                                # print(word.text)
                                newinfo.digittypewhat.append(re.findall('[\d ]+', oneword.lemma_))  # digit
                                newinfo.digittypewhere.append(re.findall('[^\d ]+', oneword.lemma_))  # string

                                # print re.findall ('[\d ]+', word.text);
                                # print re.findall ('[^\d ]+', word.text);

                            if (oneword.pos_ in ["NUM"] and oneword.is_digit == True):
                                newinfo.digittypewhat.append(oneword.lemma_)
                                # recorder.digittypewhere.append(word.head.text)
                                if (len(doc) > oneword.i + 1):
                                    newinfo.digittypewhere.append(
                                        doc.__getitem__(oneword.i + 1).lemma_)  # get next item
                                # for child in word.children:
                                #    newinfo.digittypewhere.append(child.text)
                            #####################################33
                    steplist.append(newinfo)

                examplemet = False
                for perword in doc:
                    if perword.text in ["e.g.", "E.g.", "E.G.", "example", "say"]:
                        examplemet = True

                    if examplemet & (perword.pos_ in ["NUM"]):
                        newinfo.digittypewhat.insert(0, perword.text)
                        break;

                    if examplemet & (perword.pos_ in ["NOUN"]):
                        newinfo.typewhat.insert(0, perword.text)
                        break;

                tag = False
                for oneword in doc:
                    if oneword.text in ["e.g.", "E.g.", "E.G.", "example", "say"]:
                        tag = True
                        break
                if tag == False:
                    recorder = newinfo
            kk += 1

            ###########################left out case#############################
            if (word.lemma_ in ["leave", "left"]):
                if (doc.__getitem__(kk + 1).lemma_ in ["out"]):

                    newinfo = strandform()
                    newinfo.types = "input"
                    newinfo.verb = str(word.lemma_ )
                    newinfo.step = step
                    newinfo.sent = sent
                    newinfo.sentenceid = sentid - 1
                    for everyword in doc:
                        newinfo.sentence.append(everyword.lemma_)

                    step += 1

                    newinfo.typewhat.append("")
                    # newinfo.typewhere.append(stringnounphrasemap.get(child.i))

                    for wordchild in word.subtree:
                        # if (wordchild.dep_ in ["dobj"]):
                        newinfo.typewhere.append(wordchild.lemma_)

                    newinfo.digittypewhat.append("")
                    # newinfo.typewhere.append(stringnounphrasemap.get(child.i))

                    for wordchild in word.subtree:
                        # if (wordchild.dep_ in ["dobj"]):
                        newinfo.digittypewhere.append(wordchild.lemma_)
                        # newinfo.typewhere.append(stringnounphrasemap.get(wordchild))

                    #    for (child.dep_ in [ ]):

                    steplist.append(newinfo)

    ###################33xml###############33
    # file_name=open(address+"/middleResults/nlp.xml","wb")
    #file_name = open(address + "/middleResults/nlp.xml", "w")
    doc = Document()
    root = doc.createElement("Steps")

    for step in steplist:
        child = doc.createElement("Step" + str(step.step))
        root.appendChild(child)

        sentenceele = doc.createElement("sentence")
        k = 0
        for sten in step.sentence:
            newitem = doc.createElement("item" + str(k))
            k += 1
            nodeText = doc.createTextNode(sten)
            newitem.appendChild(nodeText)
            sentenceele.appendChild(newitem)
        child.appendChild(sentenceele)

        idinfo = doc.createElement("sentenceid")
        nodeText = doc.createTextNode(str(step.sentenceid))
        idinfo.appendChild(nodeText)
        child.appendChild(idinfo)

        #######################

        ################333input
        if (step.types == "input"):
            types = doc.createElement("type")
            nodeText = doc.createTextNode("input")
            types.appendChild(nodeText)

            typetimes = doc.createElement("inputtimes")
            nodeText = doc.createTextNode(str(step.typetimes))
            typetimes.appendChild(nodeText)

            typewhere = doc.createElement("typewhere")
            i = 0
            for where in step.typewhere:
                ID = doc.createElement("ID" + str(i))
                i = i + 1
                typewhere.appendChild(ID)

                if not isinstance(where, list):
                    oneitem = doc.createElement("item0")
                    # nodeText=doc.createTextNode(unicode(where))
                    nodeText = doc.createTextNode(str(where))
                    oneitem.appendChild(nodeText)
                    ID.appendChild(oneitem)
                else:
                    k = 0
                    for itemwhere in where:
                        newitem = doc.createElement("item" + str(k))
                        k += 1
                        # nodeText=doc.createTextNode(unicode(itemwhere))
                        nodeText = doc.createTextNode(str(itemwhere))
                        newitem.appendChild(nodeText)
                        ID.appendChild(newitem)

            typewhat = doc.createElement("typewhat")
            i = 0
            for where in step.typewhat:
                ID = doc.createElement("ID" + str(i))
                i = i + 1
                typewhat.appendChild(ID)

                if not isinstance(where, list):
                    oneitem = doc.createElement("item0")
                    # nodeText=doc.createTextNode(unicode(where))
                    nodeText = doc.createTextNode(str(where))
                    oneitem.appendChild(nodeText)
                    ID.appendChild(oneitem)
                else:
                    k = 0
                    for itemwhere in where:
                        newitem = doc.createElement("item" + str(k))
                        k += 1
                        # nodeText=doc.createTextNode(unicode(itemwhere))
                        nodeText = doc.createTextNode(str(itemwhere))
                        newitem.appendChild(nodeText)
                        ID.appendChild(newitem)

            digittypewhere = doc.createElement("digittypewhere")
            i = 0
            for where in step.digittypewhere:
                ID = doc.createElement("ID" + str(i))
                i = i + 1
                digittypewhere.appendChild(ID)

                if not isinstance(where, list):
                    oneitem = doc.createElement("item0")
                    # nodeText=doc.createTextNode(unicode(where))
                    nodeText = doc.createTextNode(str(where))
                    oneitem.appendChild(nodeText)
                    ID.appendChild(oneitem)
                else:
                    k = 0
                    for itemwhere in where:
                        newitem = doc.createElement("item" + str(k))
                        k += 1
                        # nodeText=doc.createTextNode(unicode(itemwhere))
                        nodeText = doc.createTextNode(str(where))
                        newitem.appendChild(nodeText)
                        ID.appendChild(newitem)

            digittypewhat = doc.createElement("digittypewhat")
            i = 0
            for where in step.digittypewhat:
                ID = doc.createElement("ID" + str(i))
                i = i + 1
                digittypewhat.appendChild(ID)

                if not isinstance(where, list):
                    oneitem = doc.createElement("item0")
                    # nodeText=doc.createTextNode(unicode(where))
                    nodeText = doc.createTextNode(str(where))
                    oneitem.appendChild(nodeText)
                    ID.appendChild(oneitem)
                else:
                    k = 0
                    for itemwhere in where:
                        newitem = doc.createElement("item" + str(k))
                        k += 1
                        # nodeText=doc.createTextNode(unicode(itemwhere))
                        nodeText = doc.createTextNode(str(itemwhere))
                        newitem.appendChild(nodeText)
                        ID.appendChild(newitem)

            nodeText = doc.createTextNode(str(step.typetimes))
            typetimes.appendChild(nodeText)

            child.appendChild(types)
            child.appendChild(typetimes)
            child.appendChild(typewhere)
            child.appendChild(typewhat)
            child.appendChild(digittypewhere)
            child.appendChild(digittypewhat)

        ###############33click
        if (step.types == "click"):
            types = doc.createElement("type")
            nodeText = doc.createTextNode("click")
            types.appendChild(nodeText)

            clicktimes = doc.createElement("clicktimes")
            nodeText = doc.createTextNode(str(step.clicktimes))
            clicktimes.appendChild(nodeText)

            clicktype = doc.createElement("clicktype")
            nodeText = doc.createTextNode(str(step.clicktype))
            clicktype.appendChild(nodeText)

            clickwhere = doc.createElement("clickwhere")
            i = 0
            for where in step.clickwhere:
                ID = doc.createElement("ID" + str(i))
                i = i + 1
                clickwhere.appendChild(ID)

                if not isinstance(where, list):
                    oneitem = doc.createElement("item0")
                    # nodeText=doc.createTextNode(unicode(where))
                    nodeText = doc.createTextNode(str(where))
                    oneitem.appendChild(nodeText)
                    ID.appendChild(oneitem)
                else:
                    k = 0
                    for itemwhere in where:
                        newitem = doc.createElement("item" + str(k))
                        k += 1
                        # nodeText=doc.createTextNode(unicode(itemwhere))
                        nodeText = doc.createTextNode(str(itemwhere))
                        newitem.appendChild(nodeText)
                        ID.appendChild(newitem)

            child.appendChild(types)
            child.appendChild(clicktimes)
            child.appendChild(clicktype)
            child.appendChild(clickwhere)
        ################cancel
        if (step.types == "cancel"):
            types = doc.createElement("type")
            nodeText = doc.createTextNode("cancel")
            types.appendChild(nodeText)

            child.appendChild(type)

        ################rotate
        if (step.types == "rotate"):
            types = doc.createElement("type")
            nodeText = doc.createTextNode("rotate")
            types.appendChild(nodeText)

            child.appendChild(type)

        ################crate
        if (step.types == "create"):
            types = doc.createElement("type")
            nodeText = doc.createTextNode("create")
            types.appendChild(nodeText)

            createwhat = doc.createElement("createwhat")
            i = 0
            for where in step.createwhat:
                ID = doc.createElement("ID" + str(i))
                i = i + 1
                createwhat.appendChild(ID)

                if not isinstance(where, list):
                    oneitem = doc.createElement("item0")
                    # nodeText=doc.createTextNode(unicode(where))
                    nodeText = doc.createTextNode(str(where))
                    oneitem.appendChild(nodeText)
                    ID.appendChild(oneitem)
                else:
                    k = 0
                    for itemwhere in where:
                        newitem = doc.createElement("item" + str(k))
                        k += 1
                        # nodeText=doc.createTextNode(unicode(itemwhere))
                        nodeText = doc.createTextNode(str(itemwhere))
                        newitem.appendChild(nodeText)
                        ID.appendChild(newitem)

            child.appendChild(types)
            child.appendChild(createwhat)

    doc.appendChild(root)
    #doc.writexml(file_name)
    #file_name.close()

    ################allcase
    doc_allcase.appendChild(root_allcase)
    #doc_allcase.writexml(file_allcase)
    #file_allcase.close()

    verbs = []
    objects =[]
    for step in steplist:
        str1 =""
        if(step.types=="input"):
            '''print "newinf,type more times"
            print step.typetimes                       
            print "newinfo,where"
            print step.typewhere
            print "newinfo,what"
            print step.typewhat
            print "digit,where"
            print step.digittypewhere
            print "digit,what"
            print step.digittypewhat'''
            verbs.append(str(step.verb))
            #print(step.typewhere)
            for i in step.typewhere:
                if type(i) == list:
                    str1 += " ".join('%s' % id for id in i)
                else:
                    str1 += " " + str(i)
            objects.append(str1)
            #print(str1)
            sents.append(str(step.sent))
        if(step.types=="click"):
           '''print "click more times"
            print step.clicktimes
            print "click where"
            print step.clickwhere
            print "click one or not"
            print step.clicktype'''
           verbs.append(str(step.verb))
           #print(step.clickwhere)
           for i in step.clickwhere:
               if type(i) == list:
                   str1 += " ".join('%s' %id for id in i)
               else:
                   str1 += " " + str(i)
           objects.append(str1)
           sents.append(str(step.sent))
        if(step.types=="create"):
            '''print "create whatt"
            print step.createwhat'''
            verbs.append(str(step.verb))
            #print(step.createwhat)
            for i in step.createwhat:
                if type(i) == list:
                    str1 += " ".join('%s' % id for id in i)
                else:
                    str1 += " " + str(i)
            objects.append(str1)
            print(str1)
            sents.append(str(step.sent))
        if(step.types=="cancel"):
            verbs.append(str(step.verb))
            objects.append("dialog")
            sents.append(str(step.sent))
        if(step.types=="rotate"):
            #print "rotate type"
            verbs.append(" ".join(str(step.verb)))
            objects.append("screen")
            sents.append(str(step.sent))
    return verbs,objects,sents

def vector_similarity(s1, s2):
    def sentence_vector(s):
        #str = re.sub(r'(_\w)','',s)
        str = re.sub(r'(_\w)',lambda x:x.group(1)[1].upper(), s)
        words = nltk.word_tokenize(str)
        results = [w for w in words if w not in stop_words]
        v = np.zeros(300)
        for word in results:
            if word in model:
                v += model[word]
        if len(words)==0:
            length=1
        else:
            length=len(words)
        v /= length
        return v

    v1, v2 = sentence_vector(s1), sentence_vector(s2)
    #print(v1)
    #print(v2)
    #similarity =np.dot(v1, v2) / (norm(v1) * norm(v2))
    if norm(v1) ==0 or norm(v2) == 0:
        similarity = 0
    else:
        similarity = np.dot(v1, v2) / (norm(v1) * norm(v2))
    return similarity
def search_for_type(search_string ,verb, project):
    o_type = ""
    ns = 'http://schemas.android.com/apk/res/android'
    xmlns = '{http://schemas.android.com/apk/res/android}'
    #search_id = "@string/"
    search_project = 'analysistool/Work/'+project+'/'
    #search_project = 'analysistool/Work/com.newsblur/'
    root_dir = search_project + "res/values/"
    string_tree = ETree.parse(root_dir + "strings.xml")
    string_root = string_tree.getroot()
    id_tree = ETree.parse(root_dir + "ids.xml")
    id_root = id_tree.getroot()
    search_string = verb + search_string
    #print(o_type)
    max_similarity=0
    name_text=""
    for node in string_root.iter("string"):
        similarity = vector_similarity(str(node.text).lower(), search_string.lower())
        if similarity > max_similarity:
            max_similarity = similarity
            name_text = node.attrib['name']
        #if node.text == search_string:
        #    search_id += node.attrib['name']
        search_id ="@string/" + name_text
    for node in id_root.iter("item"):
        similarity = vector_similarity(str(node.text).lower(), search_string.lower())
        if similarity > max_similarity:
            max_similarity = similarity
            name_text = node.attrib['name']
        search_id = "@id/" + name_text
    #print(search_id)
    layout_dir = search_project + "res/layout/"

    for file in os.listdir(layout_dir):
        layout_tree = ETree.parse(layout_dir + file)
        layout_root = layout_tree.getroot()
        if "@id/" in search_id:
            for node in layout_root.iter():
                if ((xmlns + 'id') in node.attrib.keys()) and node.attrib[xmlns + 'id'] == search_id:
                    o_type = str(node.tag)
                    break
        if "@string/" in search_id:
            for node in layout_root.iter():
                if ((xmlns + 'text') in node.attrib.keys()) and node.attrib[xmlns + 'text'] == search_id:
                    o_type = str(node.tag)
                    if ((xmlns + 'clickable') in node.attrib.keys()) and node.attrib[xmlns + 'clickable'] == 'true':
                        o_type = "Button"
                    break
                if ((xmlns + 'hint') in node.attrib.keys()) and node.attrib[xmlns + 'hint'] == search_id:
                    o_type = str(node.tag)
                    if ((xmlns + 'clickable') in node.attrib.keys()) and node.attrib[xmlns + 'clickable'] == 'true':
                        o_type = "Button"
                    break
    layout_dir1 = search_project + "res/menu/"
    for file in os.listdir(layout_dir1):
        layout_tree = ETree.parse(layout_dir1 + file)
        layout_root = layout_tree.getroot()
        if "@id/" in search_id:
            for node in layout_root.iter():
                if ((xmlns + 'id') in node.attrib.keys()) and node.attrib[xmlns + 'id'] == search_id:
                    o_type = str(node.tag)
                    break
        if "@string/" in search_id:
            for node in layout_root.iter():
                if ((xmlns + 'text') in node.attrib.keys()) and node.attrib[xmlns + 'text'] == search_id:
                    o_type = "Menu"
                    if ((xmlns + 'clickable') in node.attrib.keys()) and node.attrib[xmlns + 'clickable'] == 'true':
                        o_type = "Button"
                    break
                if ((xmlns + 'hint') in node.attrib.keys()) and node.attrib[xmlns + 'hint'] == search_id:
                    o_type = "Menu"
                    if ((xmlns + 'clickable') in node.attrib.keys()) and node.attrib[xmlns + 'clickable'] == 'true':
                        o_type = "Button"
                    break
    if "None" == search_string:
        o_type ="None"
    if max_similarity < 0.15:
        o_type = "None"

    return o_type

def main(address,appName):
    fileObject = open(address + '/bugreport')
    #nlp = English()
    nlp = spacy.load('en')
    lineList = []
    verb_spacy = []
    verb_pick = []
    verb_final = []
    objects = []
    objects_final=[]
    o_types = []
    sentences =[]
    sents = []
    sents_final =[]
    for line in fileObject:
        s1 = str(line)
        sentlist=re.split('and|or|before|after',s1)
        lineList.extend(sentlist)
        # lineList.append(unicode(line))
        #lineList.append(str(line))
    print("read over")
    i=0
    for line in lineList:

        verb, object, sent = pickupcrital(line)
        #print(verb)
        #print(object)
        print(sent)
        verb_pick.extend(verb)
        objects.extend(object)
        sentences.extend(sent)
        for sent in nlp(line.lower()).sents:
            doc = nlp(str(sent))
            for word in doc:
                if word.pos_ in ["VERB"]:
                    verb_spacy.append(word.lemma_)
                    sents.append(str(sent))

    print("search over")
    for i in range(len(verb_spacy)):
        pos = -1
        for j in range(len(verb_pick)):
            #if verb_spacy[i] == verb_pick[j] and sents[i] == sentences[j]:
            if verb_spacy[i] == verb_pick[j]:
                pos = j
                break

        verb_final.append(verb_spacy[i])
        if pos >= 0:
            objects_final.append(objects[pos])
            sents_final.append(sentences[pos])
        else:
            #verb_final.append(verb_pick[i])
            objects_final.append("None")
            sents_final.append(sents[i])
    print("combine over")
    for i in range(len(verb_final)):
        o_type = search_for_type(objects_final[i], verb_final[i],appName)
        o_types.append(o_type)
    print("type over")
    return verb_final,objects_final,o_types,sents_final
    #o_type=search_for_type("register" ,"click", "com.newsblur")
    #print(o_type)
if __name__ == '__main__':
    address = "."
    appName = "com.newsblur"
    verb, object, o_type, sents = main(address, appName)
    db = MySQLdb.connect("localhost","root","root","test", charset='utf8')
    cursor = db.cursor()
    for i in range(len(verb)):
        cursor.execute("INSERT INTO verbs_copy(VERB,CONTENT,OBJECT,O_TYPE) VALUES('%s', '%s', '%s', '%s')" % (verb[i], sents[i], object[i], o_type[i]))
        #print("insert"+ i)
        db.commit()
    db.close()

    print("insert over")



