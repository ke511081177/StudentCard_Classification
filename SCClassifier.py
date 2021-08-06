

from tensorflow import keras
import tensorflow as tf
import cv2
import numpy as np 
import os

import time
# from loss import*
# from func import*

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="XXXXXX.json"
from google.cloud import vision
client = vision.ImageAnnotatorClient()
import jieba
import re

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)




#school = ['大','學','大學','高中','國中','國小','二專','四技','學校','私立','公立']

school = ['大','學','大學','二專','四技','學校','私立','公立']

class diplomaClassifier():

    def __init__(self):
        self.preModel = keras.models.load_model('studentCardORnot.h5')
        self.tempSchool =''
        self.folderpath = './classify'
        self.folderlist = []

    def run(self,file):
        
        try:
            start = time.time()
            path = './demo'
            self.img = cv2.imread('{}/{}'.format(path,file))
            
            """Detects text in the file."""
            
            file = re.sub('[.jpg]', '', file)
        
            temp = self.checkStudentCard()
            
            if temp == 0:
                print('NotStudentCard')
                self.Unqualified('notStudentCard',file)
                return
            self.scanText()

            # 沒有偵測到字詞
            if len(self.texts) == 0:
                
                print('NO text')
                self.Unqualified('unsorted',file)
                print("執行時間：%f 秒" % (time.time() - start))
                return 0

            text = self.texts[0].description
            text = text.replace("\n","").strip()
            text = text.replace(" ","").strip()

            jieba.load_userdict('./jiabaDictionary/school.txt')

            seg_list = jieba.cut(text)
            
            temp_text = [] 

            for i in seg_list:
                if i ==' ':
                    continue
                temp_text.append(i)

            buffer = []
            for c in school:
                for i in temp_text:   
                    if c in i:
                        buffer.append(i)
                        

            #沒有偵測到跟學校有關的字詞
            if len(buffer) == 0:

                print('school: Not Found')
                self.Unqualified('unsorted',file)
                print("執行時間：%f 秒" % (time.time() - start))
                return
                

            self.tempSchool = ''

            if len(buffer) != 0:
                for i in buffer:
                    if len(self.tempSchool) < len(i):
                        self.tempSchool = i
                
                if len(self.tempSchool) < 4:

                    print('school: Not Found')
                    self.Unqualified('unsorted',file)
                    
                else:
                    self.qualified(file,text)
            print("執行時間：%f 秒" % (time.time() - start))
        except:
            print('--ERROR--')
            print(file)

    def checkStudentCard(self):

        temp = self.img
        temp = cv2.resize(temp, (300,300))
        
        temp = temp.reshape(1,300,300,3)

        temp = self.preModel.predict(temp)
        
        if temp[0][1] > 0.7:
            temp = 1
        else:
            temp = 0
        
        return temp

    def scanText(self):
        # 編碼用於傳輸
        success, encoded_image = cv2.imencode('.jpg', self.img)
        content = encoded_image.tobytes()

        # Call API
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
    
        self.texts = response.text_annotations

    def qualified(self,file,text):

        if self.tempSchool not in self.folderlist:
            os.makedirs('./{}/{}'.format(self.folderpath,self.tempSchool))
            self.folderlist.append(self.tempSchool)

        os.makedirs('./{}/{}/{}'.format(self.folderpath,self.tempSchool,file))
        cv2.imencode('.jpg',self.img)[1].tofile('{}/{}/{}/{}.jpg'.format(self.folderpath,self.tempSchool,file,file)) 

        #print('{}/{}/{}/{}_text.txt'.format(self.folderpath,self.tempSchool,file,file))

        with open('{}/{}/{}/{}_text.txt'.format(self.folderpath,self.tempSchool,file,file), 'w',encoding="utf-8") as f:
            f.write(text) 

    def Unqualified(self,folder,file):

        os.makedirs('./{}/{}/{}'.format(self.folderpath,folder,file))
        cv2.imencode('.jpg',self.img)[1].tofile('{}/{}/{}/{}.jpg'.format(self.folderpath,folder,file,file))
        if folder == 'unsorted':
            cv2.imencode('.jpg',self.img)[1].tofile('./A/{}.jpg'.format(file))
        

if __name__ == '__main__':
    classifiler = diplomaClassifier()

    path = './demo'
    filelist = os.listdir(path)

    jieba.case_sensitive = True
    
    # classifiler.img  = cv2.imread('{}/{}'.format(path,'671.jpg'))
    # classifiler.run('671.jpg')

    for file in filelist:
        
        #classifiler.img  = cv2.imread('{}/{}'.format(path,file))
        classifiler.run(file)

