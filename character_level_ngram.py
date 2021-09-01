#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import random
from matplotlib import pyplot as plt

precision_cutoff, recall_cutoff = 0.96, 0.96

maximum_iteration = 350
min_n, max_n = 3, 7


def validate_method(dataset, mode, k=None, iterations=None):
    if mode == 'simple_split':
        
        # train = pd.read_csv('opioids_list.csv')
        # xTrain = pd.DataFrame(pd.concat([train[column].dropna().drop_duplicates() for column in train.columns])).reset_index()[0]
        # yTrain = pd.DataFrame(np.array([1] * len(xTrain)))[0]
        # xPos = xTrain
        # xNeg = []
        # test = dataset

        train = dataset
        test = pd.read_csv('opioids_list.csv', usecols=['brand_names', 'generic_names', 
                                                        'combination_opioid_prescriptions_brand_names', 
                                                        'combination_opioid_prescriptions_medications', 
                                                        'other_brand_names', 'other_medication'])
        xTest = pd.DataFrame(pd.concat([test[column].dropna().drop_duplicates() for column in test.columns])).reset_index()[0]
        yTest = pd.DataFrame(np.array([1] * len(xTest)))[0]
        
        # train, test = train_test_split(dataset, test_size=0.3, shuffle=True, random_state=123)  
        xTrain = train['description']
        yTrain = train['label']  
        xPos = train[train['label'] == 1]['description']
        xNeg = train[train['label'] == 0]['description']
        # xTest = test['description']
        # yTest = test['label']
        cl = Classifier(xTrain, yTrain, xPos, xNeg, precision_cutoff, recall_cutoff, min_n, max_n, maximum_iteration)
        tp, tn, fn, fp, tp_regex, tn_regex, fn_regex, fp_regex, precision, recall = cl_train_test(cl, xTest, yTest)
        
        write_cases(tp, tn, fn, fp, tp_regex, tn_regex, fn_regex, fp_regex, 'error_analysis_new_list_train.csv')
        
        plot_epochs(cl.epochs, cl.precision_list, cl.recall_list, precision, recall)
        
        print('precision score for simple split validation method: ', precision)
        print('recall score for simple split validation method: ', recall)   
        
    elif mode == 'kfold':
        
        train_precisions, train_recalls = [], []
        precisions, recalls, f1s, accuracies = [], [], [], []
        dataset_copy = dataset.copy()
        fold_size = len(dataset) // k
        fold_splits = []
        indexes = [i for i in range(len(dataset_copy))]
        for _ in range(k):
            fold = pd.DataFrame()
            while len(fold) < fold_size and len(dataset_copy) > 0:
                index = random.choice(indexes)
                fold = fold.append(dataset_copy.loc[index])
                dataset_copy.drop(index, inplace=True)
                indexes.remove(index)
            fold_splits.append(fold)
        
        for i in range(k):
            test = fold_splits[i]
            train = dataset[~dataset.index.isin(test.index)]
            print(test.head(5), train.head(5))
            xTrain = train['description']
            yTrain = train['label']
            xPos = train[train['label'] == 1]['description']
            xNeg = train[train['label'] == 0]['description']
            xTest = test['description']
            yTest = test['label']
            cl = Classifier(xTrain, yTrain, xPos, xNeg, precision_cutoff, recall_cutoff, min_n, max_n, maximum_iteration)
            tp, tn, fn, fp, tp_regex, tn_regex, fn_regex, fp_regex, precision, recall, f1_, accuracy = cl_train_test(cl, xTest, yTest)
            
            train_precisions.append(cl.precision)
            train_recalls.append(cl.recall)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1_)
            accuracies.append(accuracy)
        
        folds = np.array([i for i in range(k)])
        plt.figure()
        plt.scatter(folds, train_precisions, marker='+', label='Train Precision')
        plt.scatter(folds, train_recalls, marker='x', label='Train Recall')
        plt.scatter(folds, precisions, marker='^', label='Test Precision')
        plt.scatter(folds, recalls, marker='o', label='Test Recall')
        plt.xticks(np.arange(0, k, 1))
        plt.xlabel('Iterations')
        plt.ylabel('Metric Scores')
        plt.suptitle('Metric Scores for %s-fold Cross Validation Method' % k)
        plt.legend()
        plt.show()
        
        print('the average precision score for %s-fold cross validation method: ' % k, np.mean(precisions))
        print('the average recall score for %s-fold cross validation method: ' % k, np.mean(recalls))
        print('the average f1 score for %s-fold cross validation method: ' % k, np.mean(f1s))
        print('the average accuracy score for %s-fold cross validation method: ' % k, np.mean(accuracies))

    elif mode == 'mc':
        
        dir = os.getcwd()
        train_precisions, train_recalls = [], []
        precisions, recalls, f1s, accuracies = [], [], [], []
        for i in range(iterations):
            train, test = train_test_split(dataset, test_size=0.3, shuffle=True, random_state=i)
            xTrain = train['description']
            yTrain = train['label']  
            xPos = train[train['label'] == 1]['description']
            xNeg = train[train['label'] == 0]['description']
            xTest = test['description']
            yTest = test['label']
            cl = Classifier(xTrain, yTrain, xPos, xNeg, precision_cutoff, recall_cutoff, min_n, max_n, maximum_iteration)
            tp, tn, fn, fp, tp_regex, tn_regex, fn_regex, fp_regex, precision, recall, f1_, accuracy = cl_train_test(cl, xTest, yTest)
            write_cases(tp, tn, fn, fp, tp_regex, tn_regex, fn_regex, fp_regex, dir+'/results_new_2/'+'%s.csv' % i)
            
            train_precisions.append(cl.precision)
            train_recalls.append(cl.recall)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1_)
            accuracies.append(accuracy)
        
        folds = np.array([i for i in range(iterations)])
        plt.figure()
        plt.scatter(folds, train_precisions, marker='+', label='Train Precision')
        plt.scatter(folds, train_recalls, marker='x', label='Train Recall')
        plt.scatter(folds, precisions, marker='^', label='Test Precision')
        plt.scatter(folds, recalls, marker='o', label='Test Recall')
        plt.xticks(np.arange(0, iterations, 5))
        plt.xlabel('Iterations')
        plt.ylabel('Metric Scores')
        plt.suptitle('Metric Scores for %s Iterations Monte Carlo Cross Validation Method' % iterations)
        plt.legend()
        plt.show()
        
        print('the average precision score for %s iterations monte carlo cross validation method: ' % iterations, np.mean(precisions))
        print('the average recall score for %s iterations monte carlo cross validation method: ' % iterations, np.mean(recalls))   
        print('the average f1 score for %s-fold cross validation method: ' % iterations, np.mean(f1s))
        print('the average accuracy score for %s-fold cross validation method: ' % iterations, np.mean(accuracies))
        

def generate_regex(feature):
    regex = feature + r'[\s\S]*?'
    return regex

class Classifier:
    def __init__(self, x, y, xPos, xNeg, precision_cutoff, recall_cutoff, min_n, max_n, maximum_iteration):
        self.x = x
        self.y = y
        
        self.xPos = xPos
        self.xNeg = xNeg
        
        self.precision = None
        self.recall = None
        self.precision_cutoff = precision_cutoff
        self.recall_cutoff = recall_cutoff
        
        self.regex_set_pos = set()
        
        self.maximum_iteration = maximum_iteration
        self.min_n = min_n
        self.max_n = max_n
        
        self.y_pred = []
        
        self.precision_list = []
        self.recall_list = []
        
        self.epochs= None
    
    def generate_ngram(self, corpus):
        vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(self.min_n, self.max_n)) # only consider chars inside word boundary
        corpus_ngram = vectorizer.fit_transform(corpus)
        ngram_values = corpus_ngram.toarray().sum(axis=0)
        features = vectorizer.vocabulary_
        leng = len(corpus)
        counts = {}
        for k, i in features.items():
            if ' ' not in k: # ignore frequent chars that consist white spaces
                counts[k] = ngram_values[i]/leng
        counts = {k: v for k, v in sorted(counts.items(), key=lambda item: -item[1])}
        return counts
    
    def filter_keywords(self, features_pos, features_neg):
        keywords_pos = list()
        for k, f in features_pos.items():
            if k not in features_neg:
                keywords_pos.append(k)
        return keywords_pos
    
    def combine(self):
        iteration = 1
        features_pos = self.generate_ngram(self.xPos)
        features_neg = self.generate_ngram(self.xNeg) if len(self.xNeg) > 0 else []
        keywords_pos = self.filter_keywords(features_pos, features_neg)
        # keywords_pos, keywords_pos_add, keywords_neg = self.filter_keywords(features_pos, features_neg)
        # print(keywords_pos)
        
        for p in keywords_pos:
            self.regex_set_pos.add(generate_regex(p))
            # self.regex_set_neg.add(generate_regex(n))
            self.apply_regex()
            self.calc_score()
            print(self.precision, self.recall)
            self.precision_list.append(self.precision)
            self.recall_list.append(self.recall)
            flag = self.validate()
            if flag is True: 
                print('reach precision recall threshold')
                self.epochs = iteration
                return
            if iteration == self.maximum_iteration: 
                print('reach maximum iteration threshold')
                self.epochs = iteration
                return
            iteration += 1
        
    def apply_regex(self):
        y_pred = []
        for x in self.x:
            prediction = self.classify(x)
            y_pred.append(prediction)
        self.y_pred = y_pred
            
    def classify(self, x):
        pred = []
        for regex in self.regex_set_pos:
            pattern = re.compile(regex)
            pred.append(0 if pattern.search(x) is None else 1)
        prediction = pred[np.argmax(np.asarray(pred))]
        return prediction
    
    def calc_score(self):
        self.precision = precision_score(self.y_pred, self.y, average='weighted', zero_division=1)
        self.recall = recall_score(self.y_pred, self.y, average='weighted', zero_division=1)
    
    def validate(self):
        if self.precision >= self.precision_cutoff and self.recall >= self.recall_cutoff:
            return True
        return False

def cl_train_test(cl, xTest, yTest):
    cl.combine()
    
    regex_dct = {}
    # y_pred = pd.DataFrame(columns=['label'], dtype='int64')
    yPred = []
    for x in xTest:
    # for i in xTest.index:
        # x = xTest.iloc[i]
        pred = []
        regex_set = []
        for regex in cl.regex_set_pos:
            pattern = re.compile(regex)
            if pattern.search(x) is not None:
                pred.append(1)
                regex_set.append(regex)
            else:
                pred.append(0)
        regex_dct[x] = regex_set
        # for regex in cl.regex_set_neg:
        #     pattern = re.compile(regex)
        #     if pattern.search(x) is not None:
        #         pred.append(0)
        # prediction = np.bincount(pred).argmax() # majority vote
        prediction = pred[np.argmax(np.asarray(pred))] # positive as long as there is a match
        # print(i, pred)
        # y_pred.loc[i, ['label']] = int(prediction)
        yPred.append(prediction)
    
    # print(y_pred)
    # yPred = y_pred['label']
    precision = precision_score(yPred, yTest, average='weighted', zero_division=1)
    recall = recall_score(yPred, yTest, average='weighted', zero_division=1)
    f1_ = f1_score(yPred, yTest, average='weighted', zero_division=1)
    accuracy = accuracy_score(yPred, yTest)
    
    # return precision, recall, f1_, accuracy
    
    tp, tn, fn, fp = [], [], [], []
    tp_regex, tn_regex, fn_regex, fp_regex = [], [], [], []
    print(yPred[:5])
    print(yTest.head())
    print(xTest.head())
    for i in range(len(yPred)):
        if yPred[i] == 0 and yTest.iloc[i] == 1:
            fn.append(xTest.iloc[i])
            fn_regex.append('\n'.join(regex_dct[xTest.iloc[i]]) if len(regex_dct[xTest.iloc[i]])>0 else None)
        elif yPred[i] == 1 and yTest.iloc[i] == 0:
            fp.append(xTest.iloc[i])
            fp_regex.append('\n'.join(regex_dct[xTest.iloc[i]]) if len(regex_dct[xTest.iloc[i]])>0 else None)
        elif yPred[i] == 1 and yTest.iloc[i] == 1:
            tp.append(xTest.iloc[i])
            tp_regex.append('\n'.join(regex_dct[xTest.iloc[i]]) if len(regex_dct[xTest.iloc[i]])>0 else None)
        else:
            tn.append(xTest.iloc[i])
            tn_regex.append('\n'.join(regex_dct[xTest.iloc[i]]) if len(regex_dct[xTest.iloc[i]])>0 else None)
    return tp, tn, fn, fp, tp_regex, tn_regex, fn_regex, fp_regex, precision, recall, f1_, accuracy

def write_cases(tp, tn, fn, fp, tp_regex, tn_regex, fn_regex, fp_regex, file):
    df = pd.concat([pd.Series(tp, name='True Positives'), pd.Series(tp_regex, name='True Positives Matched Regular Expressions'),
                    pd.Series(tn, name='True Negatives'), pd.Series(tn_regex, name='True Negatives Matched Regular Expressions'),
                    pd.Series(fn, name='False Negatives'), pd.Series(fn_regex, name='False Negatives Matched Regular Expressions'),
                    pd.Series(fp, name='False Positives'), pd.Series(fp_regex, name='False Positives Matched Regular Expressions')], axis=1)
    df.to_csv(file)

def plot_epochs(iterations, precisions, recalls, test_precision, test_recall):
    epochs = np.array([i for i in range(iterations)])
    
    plt.figure()
    plt.scatter(epochs, precisions, marker='+', label='Train Precision')
    plt.scatter(epochs, recalls, marker='x', label='Train Recall')
    plt.axhline(y=test_precision, color='red', linestyle='-', label='Test Precision')
    plt.axhline(y=test_recall, color='green', linestyle='-', label='Test Recall')
    plt.xticks(np.arange(0, iterations, 20))
    plt.xlabel('Epochs')
    plt.ylabel('Metric Scores')
    plt.suptitle('Metric Scores for %s Epochs Trained on Old Opioids List' % iterations)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    dataset = pd.read_csv('corrected_dataset.csv')
    validate_method(dataset, 'mc', iterations=5)   
    