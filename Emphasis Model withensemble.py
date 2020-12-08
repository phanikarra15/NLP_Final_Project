#Install the transformers for using pre-trained Models
!pip install transformers 

#Importing all the necessary packages
import torch
import pickle
import math
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
import warnings
import time
import copy
import numpy as np
from earlystopping import EarlyStopping
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, log_loss

#Defining the tokenizer and pre_trained model 
#Incase of ERNIE Large Model
erine_larg_tok = AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-large-en")
erine_larg_mod = AutoModel.from_pretrained('nghuyong/ernie-2.0-large-en')

#Incase of ERNIE Normal Model
erine_small_tok = AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-en")
erine_small_mod = AutoModel.from_pretrained('nghuyong/ernie-2.0-en')

#Incase of BERT Normal Uncased Model
bertuc_small_tok = BertTokenizer.from_pretrained('bert-base-uncased')
bertuc_small_mod = BertModel.from_pretrained("bert-base-uncased")

#Incase of BERT Normal Cased Model
bertc_small_tok = BertTokenizer.from_pretrained('bert-base-cased')
bertc_small_mod = BertModel.from_pretrained("bert-base-cased")

#Incase of ROBERTa Normal Model
robert_small_tok = RobertaTokenizer.from_pretrained("roberta-base")
robert_small_mod = RobertaModel.from_pretrained('roberta-base')

# Load the train, test and dev dataset
def load_dataset(filename):
    with open(filename,'r') as fp:
        lines = [line.strip() for line in fp]
    return lines

# Getting the words, pos tags, probablities in a single list from both the Train and Dev dataset
def word_traindev_Data(data):
    wordLines = data
    words = []
    probabilities = []
    wordList = []
    pos = []
    empty = []
    for line in wordLines:
        lineSplit = line.strip().split('\t')
        if line:
            word = lineSplit[1]
            prob = lineSplit[4]
            temp = lineSplit[5]
            words.append(word)
            probabilities.append(float(prob))
            pos.append(temp)
        elif not (len(empty) and []):
            wordList.append((words, pos, probabilities))
            words = []
            probabilities = []
            pos = []
    return wordList

# Getting the words in a single list from the Test dataset
def word_test_Data(data):
    wordLines = data
    words = []
    testWord = []
    empty = []
    for line in wordLines:
        lineSplit = line.strip().split('\t')
        if line:
            word = lineSplit[1]            
            words.append(word)
        elif not len(empty):
            testWord.append(words)
            words = []       
    return testWord

# Generate separate list of words, pos and probablities for Train and Dev data
def data_preprocess_train_dev(data):
    text = []
    pos = []
    probs = []
    for i,j,k in data:
            text.append(i)
            pos.append(j)
            probs.append(k)
    return text,pos, probs

# Generate separate list of words for Test data
def data_preprocess_test(data):
    text = []
    for i in data:
            text.append(i)
    return text

# Replicating probablities for matching length incase of sub tokenized words
def prob_list(batch_data,batch_probs, tokenizer):
    pb = []
    for i,j in zip(batch_data,batch_probs):
        tp = []
        for k,l in zip(i,j):
            temp = tokenizer.tokenize(k)
            if len(temp) == 1:
                tp.append(float(l))
            if len(temp) > 1:
                for i in range(len(temp)):
                    tp.append(float(l))
        pb.append(tp)
    return pb

# Replicating feature vectors for matching length incase of sub tokenized words
def feature_list(batch_data,feature, tokenizer):
    fv = []
    for i,j in zip(batch_data,feature):
        tp = []
        for k,l in zip(i,j):
            temp = tokenizer.tokenize(k)
            if len(temp) == 1:
                tp.append(l)
            if len(temp) > 1:
                for i in range(len(temp)):
                    tp.append(l)
        fv.append(tp)
    return fv

# Generate sentence from words in dataset
def get_sentence(words, tokenizer):    
    tokenized_text = []
    for i in words:
        sent = ''
        for h in i:
            if sent == '':
                sent = sent + h
            else:
                sent = sent+ " " +h
        tokens = tokenizer.tokenize(sent)
        tid = tokenizer.encode(tokens, add_special_tokens=False)
        tokenized_text.append(tid)
    return tokenized_text

# function to pad data for equal length
def pad_func(data):
    max_len = 0
    for i in data:
        if len(i) > max_len:
            max_len = len(i)
    if type(i[0]) is list:
        padded = [i + [[0, 0, 0, 0, 0]]*(max_len-len(i)) for i in data]
    else:
        padded = [i + [0]*(max_len-len(i)) for i in data]
    return padded

#data augmentation function to randomly reverse a sentence, capitalize a word and remove a word from a sentence
def data_augment(words, probs):
    aug_word_list = []
    aug_prob_list = []
    for i in range(len(words)):
        aug_word_list.append(words[i])
        aug_prob_list.append(probs[i])
        
        if (i%2) == 0:
            temp_word = copy.copy(words[i])
            temp_word.reverse()
            aug_word_list.append(temp_word)
            
            temp_prb = copy.copy(probs[i])
            temp_prb.reverse()
            aug_prob_list.append(temp_prb)
            
        if (i%3) == 0:
            temp_word = copy.copy(words[i])
            temp_word[0] = temp_word[0].upper()
            aug_word_list.append(temp_word)
            
            temp_prb = copy.copy(probs[i])
            aug_prob_list.append(temp_prb)
            
        if (i%5) == 0:
            temp_word = copy.copy(words[i])
            temp_word.remove(temp_word[0])
            aug_word_list.append(temp_word)
            
            
            temp_prb = copy.copy(probs[i])
            temp_prb.remove(temp_prb[0])
            aug_prob_list.append(temp_prb)
                
    return aug_word_list, aug_prob_list

#create feature vector for the words based on starts with capital, full word is capital, has hashtags, 
#word can be tokenized and word that is a connector word
def feature_add(trainWords, trainTags, tokenizer):
    feature = []
    conn = ['a','an','and','the','or','but','yet','on', 'in','of','for','he','she','it','i','.','?','!','have','had','has','her','him','been']
    tags = ['NNP','VBN','NNS','NN','VB','PDT','VBD','RB','CB','VBG','CD','JJ']
    for i,k in zip(trainWords, trainTags):
        temp1 = []
        for j,l in zip(i,k):
            temp2 =[0] * 6
            if j[0].isupper():
                temp2[0] = 1
            else:
                temp2[0] = 0
            if '#' in j:
                temp2[1] = 1
            else:
                temp2[1] = 0
            if j.isupper():
                temp2[2] = 1
            else:
                temp2[2] = 0
            if len(tokenizer.tokenize(j))>1:
                temp2[3] = 1
            else:
                temp2[3] = 0
            if j.lower() not in conn:
                temp2[4] = 1
            else:
                temp2[4] = 0
            if l not in tags:
                temp2[5] = 0
            else:
                temp2[5] = 1
            temp1.append(temp2)
        feature.append(temp1)
    return feature

#function to shuffle the dataset 
def func_shuffle(tokens, probablities, feature):
    mapIndexPosition = list(zip(tokens, probablities, feature))
    np.random.shuffle(mapIndexPosition)
    tokens, probablities, feature = zip(*mapIndexPosition)
    return tokens, probablities, feature

# function to get attention mask
def gen_attention(data):
    attention_mask = []
    for i in data:
        tmp = list([1] * (np.count_nonzero(i))) + list([0] * (len(i) - (np.count_nonzero(i))))
        attention_mask.append(tmp)
    return attention_mask


#getting tokens, features and probablities
def get_parts_data(Words, Labels, tokenizer):
    tokens = get_sentence(Words, tokenizer)
    probablities = prob_list(Words,Labels, tokenizer)
    features = feature_add(Words, tokenizer)
    feature = feature_list(Words,features, tokenizer)
    tokens, probablities, feature = func_shuffle(tokens, probablities, feature)
    tokens_pad = pad_func(tokens)
    probablities_pad = pad_func(probablities)
    feature_pad = pad_func(feature)
    attention_pad = gen_attention(tokens_pad)
    return tokens_pad, probablities_pad, feature_pad, attention_pad

#Function for getting first 4 emphasized words
#Done by Phani
def finalProbs(data,values):        
    temp_list = [list(x) for x in zip(data,values)]
    sentence_list = []
    probas_list = []
    for sentences,probas in temp_list:
        sentence_list.append([[list] for list in sentences])
        probas_list.append([prob for prob in probas])

    wordsFinal = []
    probFinal = []
    temp2 = []
    for word, prob in zip(sentence_list,probas_list):
        wordList = []
        probList = []
        for i,j in zip(word,prob):
            if not(i[0].startswith("##")):
                wordList.append(i)
                probList.append(j)
            else:
                wordTemp = wordList[-1]+[i[0]]
                probTemp = probList[-1]+[j[0]]
                wordTemp = [''.join(wordTemp)]
                wordList.append(wordTemp)
                probList.append(probTemp)
                del(wordList[-2])
                del(probList[-2])
      
        for k in probList:
            if len(k) == 1:
                temp2.append(k)
            else:
                average = [np.average(k)]
                temp2.append(average)
        wordsFinal.append(wordList)
        probFinal.append(temp2)
        wordList = []
        probList = []
        temp2 = []
    return wordsFinal,probFinal

def compute_loss(i):
    wlist = []
    plist = []
    for j in i:
        wlist.append(j[0])
        plist.append(j[1])
        wtemp = []
        ptemp = []
    for i,j in sorted(zip(plist,wlist),reverse = True):
        wtemp.append(j)
        ptemp.append(i)
        
    wfinal = []
    loss = []
    finalList = []
    for i,j in zip(wtemp,ptemp):
        for k,l in zip(wtemp[1:],ptemp[1:]):
            currentWord = i[0]
            currentProb = float(j[0])
            nextprob = float(l[0])
            temp = currentProb - nextprob
            lossTemp = -max((temp),0) * math.log1p(temp)
            loss.append(lossTemp)
        wfinal.append([[currentWord],[currentProb],[np.average(loss)]])
    finalList.append(wfinal)
    
    return finalList

def final_rank(words,probs):
    loss_test = [] 
    for i,j in zip(words,probs):
        loss_temp = []
        for k,l in zip(i,j):
            if '##' in k[0]:
                loss_temp.append([k,l])
        if loss_temp is []:
            loss_temp.append('[]')
        loss_test.append(loss_temp)
    
    
    subword_dict = []
    subword_list = []
    for i in loss_test:
        empty_dict = dict.fromkeys(['Rank1','Rank2','Rank3','Rank4'])
        if (i == []):
            subword_list.append([["No subwords"]])
            subword_dict.append(empty_dict)
            continue
        else:
            if (len(i) == 1):
                subword_list.append(i)
                for a in i:
                    empty_dict['Rank1'] = a[0]
                subword_dict.append(empty_dict)
            else:
                j = compute_loss(i)
                subword_list.append(j)
                for c in j:
                    for d in c:
                        if empty_dict['Rank1'] is None:
                            empty_dict['Rank1'] = d[0]
                        elif empty_dict['Rank2'] is None:
                             empty_dict['Rank2'] = d[0]
                        elif empty_dict['Rank3'] is None:
                            empty_dict['Rank3'] = d[0]
                        else:
                            empty_dict['Rank4'] = d[0]
                subword_dict.append(empty_dict)
                
    wd = []
    for i,j in zip(words,probs):
        dic = sorted(zip(j,i),reverse=True)
        wd.append(dic)
        word_dict = []    
    for i,k in zip(wd,subword_dict):
        empty_word_dict = dict.fromkeys(['Rank1','Rank2','Rank3','Rank4'])
        for j in i:
            if '##' not in j[1][0]:
                if empty_word_dict['Rank1'] is None:
                    empty_word_dict['Rank1'] = j[1]
                elif empty_word_dict['Rank2'] is None:
                    empty_word_dict['Rank2'] = j[1]
                elif empty_word_dict['Rank3'] is None:
                    empty_word_dict['Rank3'] = j[1]
                elif empty_word_dict['Rank4'] is None:
                    empty_word_dict['Rank4'] = j[1]
        word_dict.append(empty_word_dict)
    
    
    final_word_dict = []
    for i,j in zip(subword_dict,word_dict):
        final_word_dict.append((i,j))
    return final_word_dict

#defining the model class
class Model(nn.Module):
    def __init__(self, pre_trained_model, tokenizer):
        super(Model, self).__init__()
        self.model = pre_trained_model
        self.tokenize = tokenizer
        
    def forward(self, words, labels):
        tokens, probablities, feature, attention = get_parts_data(words, labels, self.tokenize)
        output = self.model(torch.tensor(tokens), torch.tensor(attention))
        return output[0], feature, probablities

#Ensembling the different pre trained models
class Ensemble(nn.Module):
    def __init__(self, modelA, modelB, layers, dropout, no_models):
        super(Ensemble, self).__init__()
        if no_models == 1:
            self.modelA = modelA
        else:
            self.modelA = modelA
            self.modelB = modelB
        self.linear = nn.Linear(layers, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, words, labels, no_models):
        if no_models == 1:
            x, feature_vect, probablities = self.modelA(words, labels)  
        else:
            x1, feature_vect, probablities1 = self.modelA(words, labels) 
            x2, feature_vect, probablities2 = self.modelB(words, labels)
            x = torch.cat((x1, x2), dim=-1)
        final_op = torch.cat((x, torch.tensor(feature_vect)), dim=-1)
        linear_output = self.linear(final_op)
        output = self.dropout(linear_output)
        proba = self.sigmoid(output)
        return proba, probablities

# Specifying dataset file names
TRAINING_FILE = "train.txt"
DEV_FILE = "dev.txt"
TEST_FILE = "test.txt"

# Preprocessing work on the dataset 
trainText = word_traindev_Data(load_dataset(TRAINING_FILE))
testEval = word_test_Data(load_dataset(TEST_FILE))
devText = word_traindev_Data(load_dataset(DEV_FILE))

trainWords,trainTags, trainLabels = data_preprocess_train_dev(trainText)
devWords, devTags, devLabels = data_preprocess_train_dev(devText)
testWords = data_preprocess_test(testEval)

#augmenting the dataset size
trainWords, trainLabels = data_augment(trainWords, trainLabels)

tokenizer_A = bertuc_small_tok
pretrained_A = bertuc_small_mod
tokenizer_B = erine_small_tok
pretrained_B = erine_small_mod


Model_A = Model(pretrained_A, tokenizer_A)
Model_B = Model(pretrained_B, tokenizer_B)

layers = 1541
dropout = 0.5
no_models = 2

Ensemble_Model = Ensemble(Model_A, Model_B, layers, dropout, no_models)

model_path = 'emp_best_model.pth'
early_stopping = EarlyStopping(model_path ,4,True)
optimizer = optim.Adamax(Ensemble_Model.parameters(), lr=0.1)
loss_func = nn.MSELoss(reduction = 'mean')

batch = 1
folds = 2
epoch = 1


combined_data = trainWords + devWords
combined_labels = trainLabels + devLabels
data = np.asarray(combined_data)
labels = np.asarray(combined_labels)
kf = KFold(n_splits=folds, random_state=0, shuffle=True)

for epoch_num in range(epoch):
    print("\nRunning epoch ---->{}".format(epoch_num))
    count = 0
    fold_training_loss = []
    fold_validation_loss = []
    
    # Dividing data into folds
    for train_index, test_index in kf.split(data,labels):
        Ensemble_Model.train()
        count = count + 1
        train_words = data[train_index]
        dev_words = data[test_index]
        train_labels = labels[train_index]
        dev_labels = labels[test_index]
        training_loss = []
        validation_loss = []
        
        # Training the model
        for i in range(0, len(train_words), batch):
            Ensemble_Model.zero_grad()
            train_probas, train_probablities_pad = Ensemble_Model(train_words[i:i+batch], train_labels[i:i+batch], no_models)
            train_grd_truth = []
            for i in train_probablities_pad:
                p = []
                for j in i:
                    q=[]
                    q.append(j)
                    p.append(q)
                train_grd_truth.append(p)
            train_batch_loss = loss_func(train_probas, torch.tensor(train_grd_truth))
            training_loss.append(train_batch_loss.item())
            train_batch_loss.backward()
            optimizer.zero_grad()
            optimizer.step()
            
        
        #Validation Run
        with torch.no_grad():
            start_time = time.time()
            for i in range(0, len(dev_tokens), batch):
                dev_probas, dev_probablities_pad = Ensemble_Model(dev_words[i:i+batch], dev_labels[i:i+batch], no_models)
                dev_grd_truth = []
                for i in dev_probablities_pad:
                    p = []
                    for j in i:
                        q=[]
                        q.append(j)
                        p.append(q)
                    dev_grd_truth.append(p)
                dev_batch_loss = loss_func(dev_probas, torch.tensor(dev_grd_truth))
                validation_loss.append(dev_batch_loss.item())
                
        fold_training_loss.append(np.average(training_loss))
        fold_validation_loss.append(np.average(validation_loss))
    print("Training loss for 8 fold = {}".format(fold_training_loss))
    print("Validation loss for 8 fold = {}".format(fold_validation_loss))
        
    print("Epoch {} Training loss ---->{}".format(epoch_num,(np.average(fold_training_loss))))
    print("Epoch {} Validation loss ---->{}".format(epoch_num,(np.average(fold_validation_loss))))

    early_stopping(np.average(fold_validation_loss), model)
    if early_stopping.early_stop is True:
        print("Early stopping")
        break

#loading the trained model
model = torch.load(model_path)

#Calculating the loss for test data
test_loss = []
pred_prob = []
batch = 392
with torch.no_grad():
    for i in range(0, len(devWords), batch):
        test_probas, test_probablities_pad = model(devWords[i:i+batch], devTags[i:i+batch], devLabels[i:i+batch])
        test_grd_truth = []
        pred_prob = test_probas.detach().numpy()
        for i in test_probablities_pad:
            p = []
            for j in i:
                q=[]
                q.append(j)
                p.append(q)
            test_grd_truth.append(p)
        test_batch_loss = loss_func(test_probas, torch.tensor(test_grd_truth))
        test_loss.append(test_batch_loss.item())               
print("Test loss ----> {}".format(np.average(test_loss)))

test_prob = []
for w,x in zip(devWords,pred_prob):
    out = w 
    temp_ans = []
    index = 0
    for i in out:
        if (len(tokenizer.tokenize(i))) == 1:
            temp_ans.append(x[index][0])
            index = index + 1
        else:
            holder = []
            for j in range(len(tokenizer.tokenize(i))):
                holder.append(x[index][0])
                index = index + 1
            prb = np.average(holder)
            temp_ans.append(prb) 
    test_prob.append(temp_ans)

x = []
y = []
for i,j in zip(test_prob, devLabels):
    for k,l in zip(i,j):
        x.append(k)
        y.append(l)
score = mean_absolute_error(y,x)

top3_words = []
top3_ground_words = []


for pred,actual,words in zip(test_prob, devLabels, devWords):
    order_temp = [i for _,i in sorted(zip(actual,words), reverse = True)]
    top3_ground_words.append(order_temp[:5])
    order_pred_temp = [i for _,i in sorted(zip(pred,words), reverse = True)]
    top3_words.append(order_pred_temp[:5])
    
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3

def match(top3_words, top3_ground_words):
    scores = []
    topk = 5
    dataset = len(top3_words)
    for i in range(len(top3_words)):
       intersect = intersection(top3_words[i], top3_ground_words[i])
       score_temp = (len(intersect))/topk
       scores.append(score_temp)
    scores_final = np.sum(scores)
    match = scores_final/dataset
    return match

final_score = match(top3_words,top3_ground_words)
print("loss score of MAE:",score)
print("final score of the match metric:",final_score)
