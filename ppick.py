#coding=utf-8

"""
Created on Sat Apr 16 19:55:45 2016
输入文件：分词数据，标签数据，词向量
存入序列化数据：训练集、开发集、测试集 （序号、句长、标签）、词典（词词典、标签词典）
@author: feng
"""

import numpy
import cPickle as pkl

LayerSize = 100
#标注字典
label_dic = {'O':0, 'B-C':1, 'I-C':2, 'B-J':3, 'I-J':4, 'B-F':5 }
    
def getWord(file):
    words = []
    fd = open(file, "r")
    for line in fd:
        for t in line.split():
            if t not in words:
                words.append(t)
    return words

def getlabel(file):
    labels = []
    fd = open(file, "r")
    for line in fd:
        labels.append([int(x) for x in line.split()])
    return labels
    
def reduceEmbedding(file, words):
    word_embedding = []
    word_dic = {}
    index = 1;
    word_embedding.append((0.2*numpy.random.uniform(-1.0,1.0,LayerSize)).tolist())
    word_dic['#DEFAULT#'] = 0
    fd = open(file, "r")
    for line in fd:
        tokens = line.split()
        if tokens[0] in words:
            word_embedding.append([float(t) for t in tokens[1:]])
            word_dic[tokens[0]] = index
            index = index+1
            words.remove(tokens[0])
    
    for w in words:
        word_embedding.append((0.2*numpy.random.uniform(-1.0,1.0,LayerSize)).tolist())
        word_dic[w] = index;
        index = index+1

    #输出词向量文件
    numpy.save(open('emb.npy','wb'),numpy.asarray(word_embedding))
    
    print '没有词向量的词有',len(words),"个"
    return word_dic
 
def digitalCorpus(file, word_dic):   
    index = []
    ne = []
    f = open(file)
    for line in f:
        tokens = line.split()
        index.append([word_dic[t] for t in tokens])
        ne.append(len(tokens))
    return index, ne

#将语料分为训练集，开发集，测试集：11:1:3    
def load2File(index, ne, label, word_dic):
    trainlex = []
    trainne = []
    trainlabel = []
    validatelex = []
    validatene = []
    validatelabel = []
    testlex = []
    testne = []
    testlabel = []
    
    sum = len(ne)
    trainindex = sum/15 * 11
    validateindex = sum/15 * 12
    for lex, n, idx , k in zip(index, ne, label,range(sum)):
        if k <= trainindex:
            trainlex.append(lex)
            trainne.append(n)
            trainlabel.append(idx)
        elif k<= validateindex:
            validatelex.append(lex)
            validatene.append(n)
            validatelabel.append(idx)
        else:
            testlex.append(lex)
            testne.append(n)
            testlabel.append(idx)
    f = open('pnn.pkl','wb')
    pkl.dump((trainlex,trainne,trainlabel), f, -1)
    pkl.dump((validatelex,validatene,validatelabel), f, -1)
    pkl.dump((testlex,testne,testlabel), f, -1)
    
    dic = {}
    dic['words2idx'] = word_dic
    dic['labels2idx']=label_dic
    pkl.dump(dic, f, -1)
    return (trainlex,trainne,trainlabel),(validatelex,validatene,validatelabel),(testlex,testne,testlabel),dic
    
    
def preProcess(seqfile, labelfile, embfile):    
    words = getWord(seqfile)
    labels = getlabel(labelfile)
    print '去重词个数为：',len(words)
    word_dic = reduceEmbedding(embfile, words)
    print '字典大小为： ', len(word_dic)
    index, ne = digitalCorpus(seqfile, word_dic)
    print 'index == labels',len(index),',',len(labels)
    print '============加载语料结束============'
    return load2File(index, ne, labels, word_dic)
    

if __name__ == '__main__':
    words = getWord('corpus-seq-digit.txt')
    labels = getlabel('label.txt')
    print '去重词个数为：',len(words)
    word_dic = reduceEmbedding('news-digit.txt', words)
    print '字典大小为： ', len(word_dic)
    index, ne = digitalCorpus("corpus-seq-digit.txt", word_dic)
    print 'index == labels',len(index),'==',len(labels)
    load2File(index, ne, labels, word_dic)
    print 'the end'