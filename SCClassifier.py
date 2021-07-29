
#from tensorflow import keras
import cv2
import numpy as np 
import os


# from loss import*
# from func import*

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="subtle-fulcrum-319206-415ab8f59c71.json"
from google.cloud import vision
client = vision.ImageAnnotatorClient()
import jieba
import re

folderlist = []

folderpath = './classify'
path = './image'

school = ['大學','高中','國中','國小','二專','四技','學校','私立','公立']



class diplomaClassifier():
    def __init__(self):
        #self.cutModel = keras.models.load_model('unet_diploma_4.hdf5')
        #self.stampModel = keras.models.load_model('unet_stamp_rgb_300_15_binary.hdf5',custom_objects={'dice_coef_loss': dice_coef_loss})
        #self.stampModel = keras.models.load_model('unet_stamp_4.hdf5')
        self.text =[]
        self.tempSchool =''

    def run(self,file):
        self.img = cv2.imread('{}/{}'.format(path,file))
        
        #self.img = cv2.imread('./{}'.format(file))

       

        """Detects text in the file."""
        tempImg = self.img
        
        file = re.sub('[.jpg]', '', file)

        # 編碼用於傳輸
        success, encoded_image = cv2.imencode('.jpg', self.img)
        content = encoded_image.tobytes()

        # Call API
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
       
        texts = response.text_annotations

        # 沒有偵測到字詞
        if len(texts) == 0:
            
            print('NO text')
            os.makedirs('./{}/unsorted/{}'.format(folderpath,file))
            cv2.imwrite('{}/unsorted/{}/{}.jpg'.format(folderpath,file,file),tempImg)
            
            return 0

        text = texts[0].description
        text = text.replace("\n"," ").strip()

        jieba.load_userdict('./jiabaDictionary/school.txt')
        seg_list = jieba.cut(text)
        
        for i in seg_list:
            if i ==' ':
                continue
            self.text.append(i)
        print(self.text)
        buffer = []
        for c in school:
            for i in self.text:   
                if c in i:
                    buffer.append(i)
                    
        print(buffer)

        #沒有偵測到跟學校有關的字詞
        if len(buffer) == 0:
            print('{}/unsorted/{}/{}.jpg'.format(folderpath,file,file))
            os.makedirs('./{}/unsorted/{}'.format(folderpath,file))
            cv2.imwrite('{}/unsorted/{}/{}.jpg'.format(folderpath,file,file),tempImg)
            return 0
        
        if len(buffer) != 0:
            for i in buffer:
                if len(self.tempSchool) < len(i):
                    self.tempSchool = i
            
            if len(self.tempSchool) < 4:
                print('school: Not Found')

                os.makedirs('./{}/unsorted/{}'.format(folderpath,file))
                cv2.imwrite('{}/unsorted/{}/{}.jpg'.format(folderpath,file,file),tempImg)
           
            else:
                
                if self.tempSchool not in folderlist:
                    os.makedirs('./{}/{}'.format(folderpath,self.tempSchool))
                    folderlist.append(self.tempSchool)
                    
                print('School: ',self.tempSchool)

                # 中文路徑要解碼
                cv2.imencode('.jpg',tempImg)[1].tofile('classify/{}/{}.jpg'.format(self.tempSchool,file)) 

    


if __name__ == '__main__':
    classifiler = diplomaClassifier()

    path = './image'
    filelist = os.listdir(path)

    jieba.case_sensitive = True
    classifiler.run('152.jpg')
    # for file in filelist:
    #     #print(file)
    #     classifiler.img  = cv2.imread('{}/{}'.format(path,file))
    #     classifiler.run(file)

