#!coding=utf-8
import numpy
import time
import sys
import subprocess
import os
import random

import ppick as pp
from is13.rnn.jordan import model
from is13.metrics.accuracy import conlleval
from is13.utils.tools import shuffle, minibatch, contextwin

segfile = 'corpus/corpus-seq.txt'
labelfile = 'corpus/label.txt'
embfile = 'corpus/emb.txt'

def callRNN():

    s = {'reload':False,
         'model':'the path of the model',
         'isemb':True,
         'lr':0.0627142536696559,
         'verbose':1,
         'decay':True, # decay on the learning rate if improvement stops
         'win':5, # number of words in the context window
         'bs':9, # number of backprop through time steps
         'nhidden':100, # number of hidden units
         'seed':345,
         'emb_dimension':100, # dimension of word embedding
         'nepochs':20}
         
    
    #获取当前文件名
    folder = os.path.basename(__file__).split('.')[0]
    if not os.path.exists(folder): os.mkdir(folder)

    # load the dataset  训练集、开发集、测试集、词典
    train_set, valid_set, test_set, dic = pp.preProcess(segfile, labelfile, embfile)

    #train_set, valid_set, test_set, dic = load.atisfold(s['fold'])

    # 字典中存在labels字典和词典 词-》编号   编号-》词
    idx2label = dict((k,v) for v,k in dic['labels2idx'].iteritems())
    idx2word  = dict((k,v) for v,k in dic['words2idx'].iteritems())
    
    #对同一个文件进行处理，处理完成后进行切分，现在没做的
    #数据集中包括编号、每行个数、编号， 训练集4:1切分为训练和开发
    train_lex, train_ne, train_y = train_set
    valid_lex, valid_ne, valid_y = valid_set
    test_lex,  test_ne,  test_y  = test_set

    #vocsize = len(set(reduce(\
    #                   lambda x, y: list(x)+list(y),\
    #                   train_lex+valid_lex+test_lex)))
    #分类个数，一共多少种类，这个可以直接赋值的
    nclasses = len(idx2word)
    
    #句子数，训练语料的训练句子，用于对句子进行遍历，把握进度
    nsentences = len(train_lex) 

    # instanciate the model
    numpy.random.seed(s['seed'])
    random.seed(s['seed'])

    #初始化模型参数
    print 'init model'
    rnn = model(    nh = s['nhidden'],
                    nc = nclasses,
                    ne = 1,
                    isemb = s['isemb'],
                    de = s['emb_dimension'],
                    cs = s['win'] )

    if s['reload']:
        print 'load model'
        rnn.load(s[model])
    # train with early stopping on validation set
    best_f1 = -numpy.inf
    s['clr'] = s['lr']
    print 'start train'
    for e in xrange(s['nepochs']):
        # shuffle
        shuffle([train_lex, train_ne, train_y], s['seed'])
        s['ce'] = e
        tic = time.time()

        for i in xrange(nsentences):
            cwords = contextwin(train_lex[i], s['win'])
            words  = map(lambda x: numpy.asarray(x).astype('int32'),\
                         minibatch(cwords, s['bs']))
            labels = train_y[i]

            for word_batch , label_last_word in zip(words, labels):
                rnn.train(word_batch, label_last_word, s['clr'])  #开始训练
                rnn.normalize()

            if s['verbose']:
                print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./nsentences),'completed in %.2f (sec) <<\r'%(time.time()-tic),
                sys.stdout.flush()
            
        # evaluation // back into the real world : idx -> words
        #通过开发集进行调参，主要调节学习率

        #对测试集进行测试，并将结果转化为字母标签
        predictions_test = [ map(lambda x: idx2label[x], \
                             rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32')))\
                             for x in test_lex ]

        #将test_y的值使用字母标签进行代替
        groundtruth_test = [ map(lambda x: idx2label[x], y) for y in test_y ]

        #进test_lex使用词本身代替
        words_test = [ map(lambda x: idx2word[x], w) for w in test_lex]

        #对开发集结果进行测试，并将结果转化为字母标签
        predictions_valid = [ map(lambda x: idx2label[x], \
                             rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32')))\
                             for x in valid_lex ]

        #将开发集标签使用字母标签替换
        groundtruth_valid = [ map(lambda x: idx2label[x], y) for y in valid_y ]

        #将valid_lex使用词替换
        words_valid = [ map(lambda x: idx2word[x], w) for w in valid_lex]

        # evaluation // compute the accuracy using conlleval.pl
        # 调用conlleval.pl，对test和valid数据集进行结果分析，并将结果进行保存
        res_test  = conlleval(predictions_test, groundtruth_test, words_test, folder +'/test'+str(e)+'.txt')
        res_valid = conlleval(predictions_valid, groundtruth_valid, words_valid, folder + '/valid'+str(e)+'.txt')

        #保存模型
        if not os.path.exists('result'): os.mkdir('result')
        rnn.save('result/'+folder+str(e))

        #对测试集的F值进行比较
        print '第',e,'次迭代的F值为：',res_test['f1'],'开发集F值为',res_valid['f1']
        if res_valid['f1'] > best_f1:            
            best_f1 = res_valid['f1']
            if s['verbose']:
                print 'NEW BEST: epoch', e, 'valid F1', res_valid['f1'], 'best test F1', res_test['f1'], ' '*20
            s['vf1'], s['vp'], s['vr'] = res_valid['f1'], res_valid['p'], res_valid['r']
            s['tf1'], s['tp'], s['tr'] = res_test['f1'],  res_test['p'],  res_test['r']
            s['be'] = e
            #开启子线程执行mv命令，其实就是改名
            subprocess.call(['mv', folder + '/test'+str(e)+'.txt', folder + '/best.test'+str(e)+'.txt'])
            subprocess.call(['mv', folder + '/valid'+str(e)+'.txt', folder + '/best.valid'+str(e)+'.txt'])
        else:
            print ''
        
        # learning rate decay if no improvement in 10 epochs
        if s['decay'] and abs(s['be']-s['ce']) >= 5: 
            s['clr'] *= 0.5
            print '学习率修改为=',s['clr'] 
        if s['clr'] < 1e-5: break

    print 'BEST RESULT: epoch', e, 'valid F1', s['vf1'], 'best test F1', s['tf1'], 'with the model', folder

if __name__ == '__main__':
    callRNN()