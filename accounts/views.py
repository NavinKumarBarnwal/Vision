from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.contrib import messages
from django.contrib.auth.models import User,auth

# Create your views here.

def login(request):
    if request.method == 'POST':
        username=request.POST['username']
        password=request.POST['pass']

        user=auth.authenticate(username=username,password=password)

        if user is not None:
            auth.login(request,user)
            return redirect('home')

        else:
            messages.error(request,'Invalid Login Credentials, Please Try Again!')
            return redirect('home')
        return redirect('/')
    else:
        return render(request,'login.html')

def register(request):

    if request.method == 'POST':
        username=request.POST['username']
        address=request.POST['address']
        password1=request.POST['pass']
        password2=request.POST['confpass']
        
        if password1==password2:
            if User.objects.filter(username=username).exists():
                messages.error(request,'User Name already exists, Please Try Again!')
                return redirect('home')
            else:
                user=User.objects.create_user(username=username,first_name=address,password=password1)
                user.save()
                messages.success(request,'User Succesfully Registered, Please Sign In!')
                return redirect('home')
            
        else:
            messages.error(request,'Entered Passwords do not match, Please Try Again!')
            return redirect('home')
        return redirect('/')

    else:
        return render(request,'register.html')


def logout(request):
    auth.logout(request)
    return redirect('/')