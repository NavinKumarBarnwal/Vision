#feature-extractor
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
import numpy as np
from keras import backend as K
from PIL import Image, ImageOps
class FeatureExtractor:
    def __init__(self):
        K.clear_session()
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    def extract(self, img):  
        img = img.resize((224, 224))  
        img = img.convert('RGB')  
        x = image.img_to_array(img)  
        x = np.expand_dims(x, axis=0)  
        x = preprocess_input(x)

        feature = self.model.predict(x)[0]  
        return feature / np.linalg.norm(feature)  

#offline
import glob
import os
import pickle

#from feature_extractor import FeatureExtractor

fe = FeatureExtractor()

for img_path in sorted(glob.glob('static/img1/*.jpg')):
    print(img_path)
    img = Image.open(img_path)  # PIL image
    feature = fe.extract(img)
    feature_path = 'static/feature/' + os.path.splitext(os.path.basename(img_path))[0] + '.pkl'
    pickle.dump(feature, open(feature_path, 'wb'))

#server1
from datetime import datetime

from .form import ImageFileUploadForm
from django.core.files.storage import FileSystemStorage    


from django.shortcuts import render,redirect
from .models import Product,UserProduct
from django.contrib import messages
from django.http import JsonResponse,HttpResponse
import math
from django.urls import resolve
from django.template import *
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
global tagp,prev
tagp=[]
prev=[]

def a1(request):
    return render(request,'1.html')
def a2(request):
    return render(request,'2.html')

def a3(request):
    return render(request,'3.html')
def a4(request):
    return render(request,'4.html')
def a5(request):
    return render(request,'5.html')
def a6(request):
    return render(request,'6.html')
def a7(request):
    return render(request,'7.html')
def a8(request):
    return render(request,'8.html')
def a9(request):
    return render(request,'9.html')
def a10(request):
    return render(request,'10.html')
def a11(request):
    return render(request,'11.html')
def a12(request):
    return render(request,'12.html')
def a13(request):
    return render(request,'13.html')
def a14(request):
    return render(request,'14.html')
def a15(request):
    return render(request,'15.html')
def a16(request):
    return render(request,'16.html')
def a17(request):
    return render(request,'17.html')
def a18(request):
    return render(request,'18.html')
def a19(request):
    return render(request,'19.html')
def a20(request):
    return render(request,'20.html')
def a21(request):
    return render(request,'21.html')
def a22(request):
    return render(request,'22.html')
def a23(request):
    return render(request,'23.html')
def a24(request):
    return render(request,'24.html')
def a25(request):
    return render(request,'25.html')
def a26(request):
    return render(request,'26.html')
def a27(request):
    return render(request,'27.html')
def a28(request):
    return render(request,'28.html')
def a29(request):
    return render(request,'29.html')
def a30(request):
    return render(request,'30.html')

def chat1(request):
    if request.method == 'POST':
        fe = FeatureExtractor()
        features=[]
        img_paths=[]
        for feature_path in glob.glob("static/feature/*"):
            features.append(pickle.load(open(feature_path, 'rb')))
            img_paths.append('static/img1/' + os.path.splitext(os.path.basename(feature_path))[0] + '.jpg')
        form = ImageFileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            x=form.save(commit=False)
            img=x.photo
            fs=FileSystemStorage()
            filename = fs.save(img.name, img)
            uploaded_file_url = fs.url(filename)
            img=Image.open(img.path)
            query = fe.extract(img)
            dists = np.linalg.norm(features - query, axis=1)  # Do search
            ids = np.argsort(dists)[:10] # Top 10 results
            scores = [(dists[id], img_paths[id]) for id in ids]
            ids = list(ids)
            y=""
            for i in ids:
                y+=(str(i)+" ")
            yu="http://vision.com/search/?usinput="+"+".join(y.split())
            #print(form)
            #data=request.POST.copy()
            #ph=data.get('photo')
            #data1={'form':ph}
            #dics={'error': False, 'message': 'Uploaded Successfully','form':form}
            #return HttpResponse(json.dumps(data1))
            return JsonResponse({'error': False, 'message': 'Uploaded Successfully','form':uploaded_file_url,'yu':yu})
        else:
            return JsonResponse({'error': True, 'errors': form.errors})
    else:
        name=request.user.username
        ui=request.GET['name']
        prev.append(ui)
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
                yu="http://vision.com/search/?usinput="+"+".join(ui.split())

            elif tag=="product":
                yu="http://vision.com/search/?usinput="+"+".join(ui.split())      

            elif tag=="showcart":
                yu="http://vision.com/shopping-cart/"

            elif tag=="checkout":
                yu="http://vision.com/check-out/"
            elif tag=="order":
                yu="http://vision.com/check-out/"
            elif tag=="explore":
                x=ui.split()[-1]
                yu="http://vision.com/"+x+"/"
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
                yu="http://vision.com/shopping-cart/"
            else:
                yu="Please Enter Relevant Product IDs."    
        elif len(tagp)>=2 and tagp[-1]=="explore":
            k=prev[-2]
            k=(k.split()[-1])
            x=ui.split()
            up=UserProduct.objects.all()
            
            if ("".join(k)).isdigit():
                c=0
                for j in up:
                    if int(k)==j.pid:
                        c+=1
                    if c==0:
                        y=UserProduct.objects.create(name=name,pid=int(i))
                        y.save()    
                yu="http://vision.com/shopping-cart.html/"
            else:
                yu="Please Enter Relevant Product IDs."    
            
        else:
            yu="Please Enter Relevant Keywords."    
    return HttpResponse(yu)   
      
def home(request):
    nslides=(n//3)+math.ceil(n/3-n//3)
    form = ImageFileUploadForm()
    params={'no_of_slides':nslides,'range':range(1,nslides), 'product':products ,'form': form}
    return render(request,'new index1.html',params)


def search(request):
    ui=request.GET['usinput']
    xx=[]
    responses=""
    o="".join(ui.split())
    if o.isdigit():
        k=[]
        for i in ui.split():
            k.append(int(i))
    else:
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
    form = ImageFileUploadForm()
    params={'pro':p1,'form':form}
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
    form = ImageFileUploadForm()
    params={'product':products,'sum':sum,'form':form}
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
    form = ImageFileUploadForm()
    params={'product':products,'sum':sum,'form':form}
    return render(request,'check-out.html',params)


def delete1(request):
    name=request.user.username
    x = UserProduct.objects.filter(name=name)
    x.delete()
    messages.success(request,'Order Placed Successfully!')
    return redirect('home')

def faq(request):
    return render(request,'faq.html')
