import pandas as pd
import numpy as np
import string 
import re
import warnings
warnings.filterwarnings("ignore")
import nltk
from typing import List
import os
from collections import Counter
import jpype 
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java,JPackage


        
#Zemberek kütüphanesinin çalıştırılması
ZEMBEREK_PATH = r"C:\jar\zemberek-full.jar"
DATA_PATH ="C:\data"

if not jpype.isJVMStarted():
    jpype.startJVM(jpype.getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH), convertStrings=False)

# Check if JVM is already started
else:
    print("JVM is already started")


TurkishTokenizer: JClass = JClass("zemberek.tokenization.TurkishTokenizer")
TurkishMorphology: JClass = JClass("zemberek.morphology.TurkishMorphology")
TurkishSentenceNormalizer: JClass = JClass(
    "zemberek.normalization.TurkishSentenceNormalizer"
)
Paths: JClass = JClass("java.nio.file.Paths")
morphology = TurkishMorphology.createWithDefaults()
tokenizer = TurkishTokenizer.DEFAULT
normalizer = TurkishSentenceNormalizer(
    TurkishMorphology.createWithDefaults(),
    Paths.get(str(os.path.join(DATA_PATH, "normalization"))),
    Paths.get(str(os.path.join(DATA_PATH, "lm","lm.2gram.slm"))),
)

"----Loading the dataset---"
df = pd.read_excel(r"C:\YouTubeComments_Analyasis_Project\Datasets\Datasets_afterPreprocessing\16.Normalization_RemovalPunc_Lemma_Stopword(1111).xlsx")


"----------------------Lemmatization-------------------------"
def Lemmatization(text):
    analysis: java.util.ArrayList = morphology.analyzeAndDisambiguate(str(text)).bestAnalysis()
    token = str(text).split()
    pos = []
    
    for index, i in enumerate(analysis):
        if index < len(token):
            if str(i.getLemmas()[0]) == "UNK":
                pos.append(token[index])
            else:
                pos.append(str(i.getLemmas()[0]))
        else:
            pos.append(token[-1])
    
    return pos


"--------------------Normalization-------------------------------------------------"

def normalize_text(text):
    normalized_text = normalizer.normalize(JString(str(text)))
    return str(normalized_text)

"------------------Removal Stopword -------------------------------------------"

stopword = set( open(str(os.path.join(DATA_PATH, "stop-words.tr.txt"))))
def removal_stopwords (text):
    return " ".join([kelime for kelime in str(text).split() if kelime not in stopword])


"--------Removal Emoticons,punctation, numbers, char -------------------------------"

punctation = string.punctuation #punctuation ='''!()-[]{};':'"\,<>./?@#$%^&*_~'''
whitelist = set('abcçdefgğhıijklmnoöpqrsştuüvwxyz ABCÇDEFGĞHIİJKLMNOÖPQRSŞTUÜVWXYZ') #Remove all non-turkish characters
def dataCleaning(text):
    text = re.sub(r"\d+", " ", str(text))
    text =  str(text).translate(str.maketrans("","",punctation))
    text =  str(text).replace("\n", " ") #satır boşluklarının kaldırılması
    #sayısal değerlerin kaldırılması
    text = filter(whitelist.__contains__,  str(text))
    text = "".join([i for i in text if (i.isalnum() or i == " ")])
    return " ".join(text.split())

#----Removal Emoticons
def removal_emoticons (text):
    emoticons = re.compile("["
                               u"\U0001F600-\U0001F64F"  
                               u"\U0001F300-\U0001F5FF"                                 
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F1E0-\U0001F1FF"  
                               u"\U00002500-\U00002BEF"                                 
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoticons.sub(r"",text)

# ----------- Find most common 100 words-----------------
def find_most_common_words(text_list: List[str], top_n: int):
    all_words = []
    for text in text_list:
        words = str(text).split()
        all_words.extend(words)
    word_counts = Counter(all_words)
    top_words = word_counts.most_common(top_n)
    top_words_dict = {word: count for word, count in top_words}
    return top_words_dict

comment_list = df["Comment"].tolist()
top_words_dict = find_most_common_words(comment_list, top_n=100)
print("Most common words:")
for word, count in top_words_dict.items():
    print(f"{word}: {count}")
    
"""
#df["removal_stopwords"] = df["Comment"].apply(removal_stopwords)
df["normalize_text"] = df["Comment"].apply(lambda text :  normalize_text(text) )
df["removal_stopwords"] = df["normalize_text"].apply(lambda text :  removal_stopwords(text) )


df["Lemmatization"] = df["removal_stopwords"].apply(lambda text :  Lemmatization(text) )
df["dataCleaning"] = df["Lemmatization"].apply(lambda text :  dataCleaning(text) )
df["removal_emoticons"] = df["dataCleaning"].apply(lambda text :  removal_emoticons(text) )




pd.concat([pd.concat([df["Cyberbullying"],df["UserName"], df["removal_emoticons"]], axis=1)]).to_excel('16.Normalization_RemovalPunc_Lemma_Stopword(1111).xlsx')
"""
#print(df.head())
jpype.shutdownJVM()
