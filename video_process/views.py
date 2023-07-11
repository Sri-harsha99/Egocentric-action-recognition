from django.shortcuts import render
from django.http import HttpResponse
import cv2
import subprocess
import os
from django.core.files.storage import FileSystemStorage
import shutil

def getFrame(sec,vidcap,count):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        print("has"+" "+str(count))
        cv2.imwrite("../LSTA-master/dataset/gtea_61/frames/S2/input/1/frame%d.jpg" % count, image)

    return hasFrames
# Create your views here.
def home(request):
    return render(request,"home.html")
def actions(request):
    return render(request,"actions.html")

def upload_video(request):
    if request.method == 'POST' and request.FILES['video']:
        video = request.FILES['video']
        fs = FileSystemStorage()
        filename = fs.save(video.name,video)
        upload_video_url = fs.url(filename)
        print('video1',upload_video_url)
        # path='C:/Users/sudar/Videos/Captures/'+video
        vidcap = cv2.VideoCapture(upload_video_url[1:])  # write the filename here
        success, image = vidcap.read()
        count = 0
        sec = 0
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        parent_dir = 'video_process/LSTA-master/dataset/gtea_61/frames/S2/input/'
        if os.path.exists(parent_dir+'1'):
            shutil.rmtree(parent_dir+'1')
        # num_folders = len(next(os.walk(parent_dir))[1])
        # print(num_folders)
        os.mkdir(parent_dir+'1')
        while success:
            # print(image)  # image frames saved as a vector
            print('hii kk')
            cv2.imwrite("video_process/LSTA-master/dataset/gtea_61/frames/S2/input/1/frame%d.jpg" % count, image)  # save frame as JPEG file
            vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            success, image = vidcap.read()
            count += 1
            sec += 0.2
        command = "python video_process/LSTA-MASTER/test_rgb.py --dataset gtea_61 --root_dir dataset --seqLen 25 --testBatchSize 32 --memSize 512 \
            --outPoolSize 100 --split 2 --checkpoint video_process/LSTA-master/gtea61/split2/rgb/model.pth.tar"     
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None, shell=True)

#Launch the shell command:
        output = process.communicate()
        encoding = 'utf-8'


        print("output=   "+str(output[0].decode(encoding)))
        output=str(output[0].decode(encoding)).split('_')[0]
        return render(request,"output.html",{"predicted":output,"url":upload_video_url})

    return render(request,"home.html")

def save_video(request):
    if request.method == 'POST':
        url = request.POST['url']
        action = request.POST['action']
        print('action',action)
        vidcap = cv2.VideoCapture(url[1:])  # write the filename here
        success, image = vidcap.read()
        count = 0
        sec = 0
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        dest_dir = 'video_process/LSTA-master/dataset/gtea_61/frames/S1/'
        dest_dir += action
        if not os.path.exists(dest_dir):            
            os.mkdir(dest_dir)
        print('baccha')
        num_folders = len(next(os.walk(dest_dir))[1])
        num_folders += 1
        # os.mkdir(dest_dir+'/'+str(num_folders))
        dest_dir += '/'+str(num_folders)
        source_dir = 'video_process/LSTA-master/dataset/gtea_61/frames/S2/input/1'
        shutil.copytree(source_dir,dest_dir)
        shutil.rmtree(source_dir)
        # while success:
        #     # print(image)  # image frames saved as a vector
        #     cv2.imwrite(parent_dir+"/"+str(num_folders)+"/frame%d.jpg" % count, image)  # save frame as JPEG file
        #     vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        #     success, image = vidcap.read()
        #     count += 1
        #     sec += 0.2
    return render(request,"thanks.html")
