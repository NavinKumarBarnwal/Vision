from django.shortcuts import render,redirect
from .models import Product,UserProduct
from django.contrib import messages
from django.http import HttpResponse
import math

# Create your views here.

products=Product.objects.all()
    
n=len(products)


import nltk
 
import sys
import time



from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

        
import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)

with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

with open("intents1.json") as file:
    data1 = json.load(file)

with open("data1.pickle", "rb") as f:
        words1, labels1, training1, output1 = pickle.load(f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load("model.tflearn")
tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training1[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output1[0]), activation="softmax")
net = tflearn.regression(net)

model1 = tflearn.DNN(net)
model1.load("model1.tflearn")

'''
except:
    model.fit(training, output, n_epoch=10000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
'''
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)
global tagp
tagp=[]



def chat1(request):
    name=request.user.username
    ui=request.GET['name']
    results = model.predict([bag_of_words(ui, words)])[0]
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    tagp.append(tag)
    if (len(tagp)<2 or tagp[-2]!="addcart") and results[results_index]>0.6:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        
        yu=random.choice(responses)
        
        if len(tagp)>=2 and tagp[-2]=="help" and tag=="product":
            yu="http://127.0.0.1:8000/search/?usinput="+"+".join(ui.split())

        elif tag=="product":
            yu="http://127.0.0.1:8000/search/?usinput="+"+".join(ui.split())      

        elif tag=="showcart":
            yu="http://127.0.0.1:8000/shopping-cart/"

        elif tag=="order":
            yu="http://127.0.0.1:8000/check-out/"


        else:
            pass
    
    elif len(tagp)>=2 and tagp[-2]=="addcart":
        x=ui.split()
        up=UserProduct.objects.all()
    
        if ("".join(x)).isdigit():
            c=0
            for i in x:
                for j in up:
                    if i==j.pid:
                        c+=1
                if c==0:
                    y=UserProduct.objects.create(name=name,pid=int(i))
                    y.save()    
            yu="http://127.0.0.1:8000/shopping-cart/"
        else:
            yu="Please Enter Relevant Product IDs."    

    
    else:
        yu="Please Enter Relevant Keywords."
        
    return HttpResponse(yu)   
      

def home(request):
    nslides=(n//3)+math.ceil(n/3-n//3)

    params={'no_of_slides':nslides,'range':range(1,nslides), 'product':products}
    return render(request,'new index1.html',params)


def search(request):
    ui=request.GET['usinput']
    xx=[]
    responses=""
    
    for i in ui.split():
        results1 = model1.predict([bag_of_words(i, words1)])[0]
        results_index1 = numpy.argmax(results1)
        tag1 = labels1[results_index1]
        if results1[results_index1]>0.8:
            for tg in data1["intents"]:
                if tg['tag'] == tag1:
                    responses = tg['responses']
        else:
            responses=[i]
        if len(responses)!=0:
            xx.append(random.choice(responses))
        else:
            xx.append(i)
    ui=" ".join(xx)
    k=[]
    for j in ui.lower().split():
        p=[]
        c=1
        for i in products:    
            if j in str(i.id) or j in i.name.lower().split() or j in i.desc.lower().split() or j in i.tag.lower().split() or j in i.size.lower().split() or j in i.color.lower().split() or j in i.company.lower().split():
                p.append(c)
            c+=1
        if len(p)!=0:
            k.append(set(p))    
    if len(k)!=0:
        k=list(set.intersection(*k))
        
    p1=Product.objects.filter(pk__in=k)
        
    params={'pro':p1}
    return render(request,'search.html',params)


def cart(request):
    #ui=request.GET['usinput']
    name=request.user.username
    up=UserProduct.objects.all()

    #x=ui.split()
    xx=[]
    
    for nam in up:
        if name==nam.name:
            xx.append(nam.pid)
    
    products=Product.objects.filter(pk__in=xx)
    sum=0
    for i in products:
        sum+=i.price
    params={'product':products,'sum':sum}
    return render(request,'shopping-cart.html',params)

def checkout(request):
    name=request.user.username
    up=UserProduct.objects.all()

    #x=ui.split()
    xx=[]
    
    for nam in up:
        if name==nam.name:
            xx.append(nam.pid)
    products=Product.objects.filter(pk__in=xx)
    sum=0
    for i in products:
        sum+=i.price
    params={'product':products,'sum':sum}
    return render(request,'check-out.html',params)


def delete1(request):
    name=request.user.username
    x = UserProduct.objects.filter(name=name)
    x.delete()
    messages.success(request,'Order Placed Successfully!')
    return redirect('home')

def faq(request):
    return render(request,'faq.html')