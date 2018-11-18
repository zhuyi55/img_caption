from django.shortcuts import render
from django.http import HttpResponse
from django.db import models
from PIL import Image
import os
from io import BytesIO
#import sample2
import im2txt.run_inference_api as imModel
import json

class Images(models.Model):
    image = models.ImageField('图片', upload_to='images', default='')

# Create your views here.
def index(request):
    return render(request, "index.html")
    
global caption_generator
caption_generator = None

#@csrf_exempt
def uploadImg(request):
    global caption_generator
    baseDir = os.path.dirname(os.path.abspath(__name__));
    jpgdir = os.path.join(baseDir, 'static');
    chkpoint_path = os.path.join(baseDir, 'im_model')
    vocab = os.path.join(chkpoint_path, 'word_counts.txt')
        
    if (caption_generator == None):
        caption_generator = imModel.run_inference_api(checkpoint_path=chkpoint_path, vocab_file=vocab)
        
    if request.method == 'POST':
        img_name_list = []
        result1_list = []
        result2_list = []
        
        imglist = request.FILES.getlist('img')
        img_cnt = len(imglist)
        for idx in range(img_cnt):
            img = imglist[idx]

            filename = os.path.join(jpgdir,img.name);
            with open(filename,'wb') as fobj:
                for chrunk in img.chunks():
                    fobj.write(chrunk);

            print (filename)
            strlist = caption_generator.generate(filename)
            #strlist = sample2.prediction()
            img_name_list.append(img.name)
            result1_list.append(strlist[0])
            result2_list.append(strlist[1])
            
            #im_pic = Image.open(BytesIO(img))
            #im_pic.save(filename)
            
        content = {
            'result1':json.dumps(result1_list),
            'result2':json.dumps(result2_list),
            'imgView': img_name_list[0]
            #'imgView': json.dumps(img_name_list)
        }
        #return render(request, 'uploadimg.html', {
        #    'result1': json.dumps(strlist[0]),
        #    'result2': json.dumps(strlist[1]),
        #    'imgView': img.name
        #    })
        return render (request, 'uploadimg.html', content)

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