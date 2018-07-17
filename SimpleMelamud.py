from nltk.parse.stanford import StanfordDependencyParser
from nltk.tokenize import word_tokenize
from nltk.tokenize.stanford import StanfordTokenizer
import re
import gensim
from numpy import dot
from numpy.linalg import norm
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer
#####################################################################################
lemmatizer = WordNetLemmatizer()
wordnet_lemmatizer = WordNetLemmatizer()
path_to_jar = 'stanford-parser-full-2017-06-09/stanford-parser.jar'
path_to_models_jar = 'stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar'

Context_Embedding = 'lexsub_context_embeddings.txt'
Word_Embedding = 'lexsub_word_embeddings.txt'
Words_File='CandidateWords'


dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
Word_Model=gensim.models.KeyedVectors.load_word2vec_format(Word_Embedding,binary=False)
Context_Model=gensim.models.KeyedVectors.load_word2vec_format(Context_Embedding,binary=False)
print('Data loading is finished!!')
###########################################################################################

f=open(Words_File,'r')
l=f.readlines()
f.close()
Candidate_Words=[]
for a in l:
    Candidate_Words.append(a.strip())


def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))


def ReturnBests(Dict,Threshold=11):

    Sorted_list=sorted(Dict, key=Dict.get, reverse=True)

    OutDict = {}
    if len(Sorted_list)<Threshold:
        for a in Sorted_list:
            OutDict[a] = Dict[a]
        return OutDict

    else:
        for a in Sorted_list[:Threshold]:
            OutDict[a] = Dict[a]
        return OutDict


def Cos_Sim(NumpyVec1,NumpyVec2):
    return dot(NumpyVec1, NumpyVec2)/(norm(NumpyVec1)*norm(NumpyVec2))

def Pcos_Sim(NumpyVec1,NumpyVec2):
    return (Cos_Sim(NumpyVec1,NumpyVec2)+1)/2

def Dependency_Based_Context_Vector_of_Sentence(Sentence,TW):
    Sentence=Sentence.lower()
    TW=TW.lower()
    Tokens = WordPunctTokenizer().tokenize(Sentence.lower())

    Sentence=re.sub(r"\d/\d", "", Sentence)
    Vector={}
    result = dependency_parser.raw_parse(Sentence)
    dep = result.__next__()
    L = list(dep.triples())
    delIndex=[]
    NewList=[]
    i=0
    # print(L)
    while(i<len(L)):
        if L[i][1] in ['prep','case']:
            dist=L[i][2]
            sourc=L[i][0]
            j=0
            while(j<len(L)):

                if L[j][0]==dist and (Tokens.index(dist[0]) > Tokens.index(sourc[0])):
                    NewList.append((sourc,L[i][1]+'_'+L[j][1],L[j][2]))
                    delIndex.append(i)
                    delIndex.append(j)
                    break
                j+=1
        i+=1
    i=0
    while(i<len(L)):
        if i not in delIndex:
            NewList.append(L[i])
        i+=1

    L=NewList[:]
    t=0
    while (t < len(Tokens)):
        v = []
        for d in L:
            if d[0][0] == Tokens[t]:
                v.append(d[1] + '_' + d[2][0].lower())

            if d[2][0] == Tokens[t]:
                v.append(d[1] + 'I_' + d[0][0].lower())
        Vector.update({Tokens[t]: v})
        t += 1

    v = {}
    for a in Vector:
        if len(Vector[a])!= 0:
            v.update({a:Vector[a]})
    Vector=[]
    W=TW.split()
    for e in W:
        Vector+=v[e]
    return Vector


def MelamudLS(Sentence,TW):
    Sentence=Sentence.lower()
    TW=TW.lower()
    C = Dependency_Based_Context_Vector_of_Sentence(Sentence,TW)
    Word_tokens=TW.split()
    if len(Word_tokens)>1:

        TW_Embed=0
        i=0
        for w in Word_tokens:
            if w.lower() in Word_Model:
                TW_Embed+=Word_Model[w.lower()]
                i+=1

        TW_Embed=TW_Embed/i
    else:
        TW_Embed = Word_Model[TW]
    Add={}
    BalAdd={}
    Mul={}
    BalMul={}
    for s in Candidate_Words:
        if not hasNumbers(s):
            s1=Cos_Sim(TW_Embed,Word_Model[s])
            sp1=Pcos_Sim(TW_Embed,Word_Model[s])
            sa=0
            sm=1
            i=0
            for c in C:
                if c in Context_Model:
                    sa+=Cos_Sim(Word_Model[s],Context_Model[c])
                    sm*=Pcos_Sim(Word_Model[s],Context_Model[c])
                    i+=1
            Add.update({s: (s1 + sa) / (i + 1)})
            Mul.update({s: (sp1 * sm) ** (1 / (1 + i))})
            if i!=0:
                BalAdd.update({s:(i*s1+sa)/(i*2)})
                BalMul.update({s:((sp1**i)*sm)**(1/(2*i))})
            else:
                i+=1
                BalAdd.update({s: (i * s1 + sa) / (i * 2)})
                BalMul.update({s: ((sp1 ** i) * sm) ** (1 / (2 * i))})
    ADD=ReturnBests(Add)
    BalADD= ReturnBests(BalAdd)
    MUL=ReturnBests(Mul)
    BalMUL=ReturnBests(BalMul)
    if TW in ADD:
        ADD.pop(TW)
    else:
        ADD.popitem()
    if TW in BalADD:
        BalADD.pop(TW)
    else:
        BalADD.popitem()
    if TW in MUL:
        MUL.pop(TW)
    else:
        MUL.popitem()
    if TW in BalMUL:
        BalMUL.pop(TW)
    else:
        BalMUL.popitem()
     

    return ADD,BalADD,MUL,BalMUL
# MelamudLS("She had to fly quickly back to Taipei to make class on Saturday afternoon .",'class')
# MelamudLS("After a surprisingly sharp widening in the U.S. August merchandise trade deficit -- $ 10.77 billion from a revised $ 8.24 billion in July and well above expectations -- and a startling 190-point drop in stock prices on Oct. 13 , the Federal Reserve relaxed short-term interest rates , knocking fed funds from around 9 % to 8 3\/4 % .",
#           'relaxed')