from django.shortcuts import render
from django.http import HttpResponse
from django.db import models
from PIL import Image
import os
from io import BytesIO
import sample2
import json

class Images(models.Model):
    image = models.ImageField('图片', upload_to='images', default='')

# Create your views here.
def index(request):
    return render(request, "index.html")
    
#@csrf_exempt
def uploadImg(request):
    if request.method == 'POST':
        img = request.FILES.get('img')
        
        baseDir = os.path.dirname(os.path.abspath(__name__));
        jpgdir = os.path.join(baseDir, 'static');
        
        filename = os.path.join(jpgdir,img.name);
        fobj = open(filename,'wb');
        for chrunk in img.chunks():
            fobj.write(chrunk);
        fobj.close();
        
        strlist = sample2.prediction()
        #im_pic = Image.open(BytesIO(img))
        #im_pic.save(filename)
        content = {
            'result2':strlist[0],
            'result2':strlist[1],
            'img': img.name
        }
        return render(request, 'uploadimg.html', {
            'result1': json.dumps(strlist[0]),
            'result2': json.dumps(strlist[1]),
            'imgView': img.name
            })
        #return render (request, 'showImg.html', content)

    return render(request, 'uploadimg.html')
    
#@csrf_exempt
def showImg(request):
    imgs = IMG.objects.all()
    content = {
        'imgs':imgs,
    }
    for i in imgs:
        print (i.img.url)
        
    return render(request, 'showimg.html', content)