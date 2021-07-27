
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



folderlist = ['unsorted']

folderpath = './classify'

school = ['大學','高中','國中','國小','二專','四技','學校','私立','公立']








class diplomaClassifier():
    def __init__(self):
        #self.cutModel = keras.models.load_model('unet_diploma_4.hdf5')
        #self.stampModel = keras.models.load_model('unet_stamp_rgb_300_15_binary.hdf5',custom_objects={'dice_coef_loss': dice_coef_loss})
        #self.stampModel = keras.models.load_model('unet_stamp_4.hdf5')
        self.text =[]
        self.tempSchool =''

    def run(self,file):
        
        path = './image'
        self.img = cv2.imread('{}/{}'.format(path,file))
        
        #self.img = cv2.imread('./{}'.format(file))

        self.imgSave = self.img


        """Detects text in the file."""
        tempImg = self.img
        
        file = re.sub('[.jpg]', '', file)


        success, encoded_image = cv2.imencode('.jpg', self.img)

        content = encoded_image.tobytes()
        image = vision.Image(content=content)

        response = client.text_detection(image=image)
        texts = response.text_annotations
        print(file)

        # 沒有偵測到字詞
        if len(texts) == 0:

            print('NOo text')
            os.makedirs('./{}/unsorted/{}'.format(folderpath,file))
            cv2.imwrite('{}/unsorted/{}/{}.jpg'.format(folderpath,file,file),tempImg)
            
            return 0

        text = texts[0].description
        text = text.replace("\n"," ").strip()

        jieba.load_userdict('./jiabaDictionary/school.txt')
        seg_list = jieba.cut(text)
    
        seg = []

        for i in seg_list:
            if i ==' ':
                continue
            seg.append(i)

        buffer = []

        for c in school:
            for i in seg:   
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
            print(self.tempSchool,'!')
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

    
    def stampCut(self):
        
        
        
        self.img = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        rimg = cv2.resize(self.img, (256, 256))


        binary = np.reshape(rimg,(1,)+rimg.shape)
        
        mask = self.stampModel.predict(binary)
        
        mask = np.stack((mask,)*3, axis=-1)
        
        mask = np.reshape(mask,(256,256,3))
        
        mask = cv2.resize(mask,(self.img.shape[1],self.img.shape[0]))
        
        mask = mask[:,:,0]

        mask = np.uint8(mask)
        
        self.img = cv2.add(self.img, np.zeros(np.shape(self.img), dtype=np.uint8), mask=mask)
        self.img = cv2.cvtColor(self.img,cv2.COLOR_RGB2BGR)
    




        
    
jieba.case_sensitive = True

if __name__ == '__main__':
    classifiler = diplomaClassifier()

    path = './image'
    filelist = os.listdir(path)

    jieba.case_sensitive = True
    #classifiler.run('temp.jpg')
    for file in filelist:
        print(file)
        classifiler.img  = cv2.imread('{}/{}'.format(path,file))
        classifiler.run(file)

