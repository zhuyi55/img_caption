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
    
global caption_generator_V3
global caption_generator_V4
global caption_generator_V2

caption_generator_V3 = None
caption_generator_V4 = None
caption_generator_V2 = None

#@csrf_exempt
def uploadImg(request):
    global caption_generator_V4
    global caption_generator_V3
    global caption_generator_V2
    
    baseDir = os.path.dirname(os.path.abspath(__name__));
    jpgdir = os.path.join(baseDir, 'static');
    chkpoint_path = os.path.join(baseDir, 'im_model')
    vocab = os.path.join(chkpoint_path, 'word_counts.txt')
        
    if (caption_generator_V4 == None):
        v4_path = os.path.join(chkpoint_path, 'v4')
        caption_generator_V4 = imModel.run_inference_api(checkpoint_path=v4_path, vocab_file=vocab, cnn_model='InceptionV4')
    if (caption_generator_V2 == None):
        v2_path = os.path.join(chkpoint_path, 'v2')
        caption_generator_V2 = imModel.run_inference_api(checkpoint_path=v2_path, vocab_file=vocab, cnn_model='InceptionResnetV2')
        
    if (caption_generator_V3 == None):
        v3_path = os.path.join(chkpoint_path, 'v3')
        caption_generator_V3 = imModel.run_inference_api(checkpoint_path=v3_path, vocab_file=vocab, cnn_model='InceptionV3')
        
    if request.method == 'POST':
        img_name_list = []
        strlist_V3 = []
        strlist_V4 = []
        strlist_V2 = []
        
        model_list = request.POST.getlist('selMod')

        imglist = request.FILES.getlist('img')
        if len(imglist) == 0:
            return render(request, 'uploadimg.html')
        img_cnt = len(imglist)
        for idx in range(img_cnt):
            img = imglist[idx]

            filename = os.path.join(jpgdir,img.name);
            with open(filename,'wb') as fobj:
                for chrunk in img.chunks():
                    fobj.write(chrunk);

            if "InceptionV3" in model_list:
                strlist_V3 = caption_generator_V3.generate(filename)
            if "InceptionV4" in model_list:
                strlist_V4 = caption_generator_V4.generate(filename)
            if "InceptionResnetV2" in model_list:
                strlist_V2 = caption_generator_V2.generate(filename)
                
            #strlist = sample2.prediction()
            img_name_list.append(img.name)
            #result1_list.append(strlist[0])
            #result2_list.append(strlist[1])
            
            #im_pic = Image.open(BytesIO(img))
            #im_pic.save(filename)
            
        content = {
            'result_v3':json.dumps(strlist_V3),
            'result_v4':json.dumps(strlist_V4),
            'result_v2':json.dumps(strlist_V2),
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