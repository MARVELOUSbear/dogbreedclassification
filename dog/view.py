from django.http import HttpResponse

# from django.http import HttpResponse
from django.shortcuts import render
from dog.models import user

def hello(request):
    context = {}
    context['hello'] = 'Hello World!'
    return render(request, 'hello.html', context)

def show(request):
    return  render(request,"addsomedog/show2.html")

def index(request):
    return render(request, "cover/index.html")

def save(request):
    email = request.GET['email']
    password2 = request.GET['password']
    test1 = user(mail=email,password=password2)
    test1.save()
    return HttpResponse("<p>注册成功！</p>")

def register(request):
    return render(request, "register/register.html")

def signin(request):
    return render(request, "signin/index.html")

def userlogin(request):
    list = user.objects.all()
    email = request.GET['email']
    password2 = request.GET['password']
    response2 = user.objects.filter(mail=email,password=password2)
    if len(response2)!=0:
        return render(request,"addsomedog/show2.html")
    else :
        return HttpResponse("<p>密码错误</p>")