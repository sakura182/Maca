from bert_serving.client import BertClient
import pandas as pd
import numpy as np
import re
from sklearn import model_selection,svm,naive_bayes,metrics
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.externals import joblib
from sklearn.metrics import f1_score

def processstr(a):
    urls=re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', a)
    for url in urls:
        a=a.replace(url,'URL')
    if(a=='None'):
        a='None'
    return a

def process_data(file_name):
    df = pd.DataFrame(pd.read_csv(file_name))
    verb=[]
    text=[]
    object=[]
    o_type=[]
    label=[]
    for i in range(df.shape[0]):
        verb.append(df.iloc[i,0])
        text.append(processstr(df.iloc[i,1]))
        object.append(df.iloc[i,2])
        o_type.append(df.iloc[i,3])
        label.append(df.iloc[i,4])
    #return text,np.array(label)
    return verb,text,object,o_type,np.array(label)

def getClassWeights(label,numclass=2):
    w=[]
    for i in range(numclass):
        w.append(np.sum(label==(i+1)))
    w=np.array(w)
    w=1./np.log(w)
    #w=10./w
    return {i+1:w[i] for i in range(numclass)},w




verb,text,object,o_type,label=process_data('data.csv')

print('Encoding...')

#text=text[:92]+text[114:]
#label=np.concatenate((label[:92],label[114:]),axis=0)


bc=BertClient(ip='localhost',check_version=False, check_length=False)


vec0=np.array(bc.encode(verb))
vec1=np.array(bc.encode(text))
vec2=np.array(bc.encode(object))
vec3=np.array(bc.encode(o_type))

vec = np.hstack((vec0,vec1,vec2,vec3))
print(vec.shape)
print('Training...')


kf=model_selection.KFold(n_splits=10,shuffle=True,random_state=8)
kf_index=0

a,b=0,0
a1=0,0
res=[]

for train_index,valid_index in kf.split(vec):

    train_x,train_y=vec[train_index],label[train_index]
    valid_x,valid_y=vec[valid_index],label[valid_index]
    wc,tw=getClassWeights(np.array(train_y),5)
    #wc,tw=getClassWeights(np.array(train_y),3)
    #print(wc)
    clf = LogisticRegression(random_state=7,dual=False,class_weight=wc, multi_class='multinomial',solver='newton-cg')#class_weight=None
    clf.fit(train_x,train_y)
    joblib.dump(clf,'./models/model'+str(kf_index)+'.m')
    kf_index+=1
    print('-'*80)
    print(kf_index,'Result:')
    predictions = clf.predict(valid_x)
    prob = clf.predict_proba(valid_x)
    #print(predictions)
    #print(prob)
    print(metrics.accuracy_score(valid_y,predictions))
    print(f1_score(valid_y,predictions,average='macro'))
    print(classification_report(valid_y, predictions,digits=3 ))
    for i in range(valid_y.shape[0]):
        if(valid_y[i]!=predictions[i]):
            #print(valid_y[i],predictions[i],valid_index[i])
            res.append([valid_y[i],predictions[i],valid_index[i],prob[i]])
    a1+=f1_score(valid_y,predictions,average='macro')
    a+=np.sum(predictions==valid_y)
    b+=valid_y.shape[0]
print('-'*80)
print('Final')
print(a*1.0/b)
print(a1*1.0/10)