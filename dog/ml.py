from django.http import HttpResponse
from django.shortcuts import render_to_response
import time
from PIL import Image
import code.DogClassification
# 接收请求数据




def search(request):
    request.encoding='utf-8'
    if 'q' in request.GET:
        message = '你搜索的内容为: ' + request.GET['q']
    else:
        message = '你提交了空表单'
    return HttpResponse(message)


def updateInfo(request):

        print(1)
        photo = request.FILES['photo']

        if photo:
            phototime = request.user.username + str(time.time()).split('.')[0]
            photo_last = str(photo).split('.')[-1]
            photoname = './templates/Images/%s.%s' % (phototime, photo_last)
            img = Image.open(photo)
            img.save( photoname)
            print(photoname)
            count = 1
            dm = code.DogClassification.DogClassification()
            ans = dm.predictNewPic(photoname)
            return HttpResponse(ans)