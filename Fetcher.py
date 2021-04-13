import os
import numpy as np
import cv2
import tensorflow as tf
from imutils import paths
import math
import random
import Config
import pickle
import glob
from moviepy.editor import VideoFileClip
from sklearn.utils import shuffle

class DataFetcher(object):
    def __init__(self):

        self.trainVideos  = [];
        self.trainLabels = [];

        self.validVideos = [];
        self.validLabels = [];

        self.loaded = False;

        self.trainIdx = 0;
        self.validIdx = 0;

        self.trainDataSize = 0;
        self.validDataSize = 0;

        self.labelDicrionary = {
                "adm": 0,
                "amu": 1,
                "ten": 2,
                "ang": 3,
                "dis": 4,
                "des": 5,
                "pri": 6,
                "anx": 7,
                "int": 8,
                "irr": 9,
                "joy": 10,
                "con": 11,
                "fea": 12,
                "ple": 13,
                "rel": 14,
                "sur": 15,
                "sad": 16,
            };
    
    def processAndSave(self):
        videoPaths = glob.glob(Config.BASE_PATH + "GEMEP_Coreset_Full Body" + "/*.avi");
        

        fileCount =0;

        translations = [[25,25],[25,-25],[-25,25],[-25,-25]];
        rotations = [];
        for i in range(-30,30):
            rotations.append(i);

        gaussianFilters = [1,2,3,4,5,6,7,8,9,10]

        brightness = [];
        for b in range(-50,50,2):
            brightness.append(b);

        totalFiles = len(videoPaths * ((len(rotations) + len(translations)+len(gaussianFilters)+len(brightness))));

        for vid in videoPaths:
            frames = np.zeros((len(rotations) + len(translations)+len(gaussianFilters)+len(brightness)
            ,200,Config.VIDEO_WIDTH,Config.VIDEO_HEIGHT,1),dtype=np.float);
            
            frameCounter = 0;
        

            lbl = self.labelDicrionary[vid.split('\\')[-1].split('_')[0][3:]];
            label = np.zeros(17);
            label[lbl]=1;
            videoName = vid.split('\\')[-1];

            clip = cv2.VideoCapture(os.path.sep.join([Config.BASE_PATH,"GEMEP_Coreset_Full Body",videoName]));

            while True:
                
                (grabbed,frame) = clip.read();
                counter = 0;

                if grabbed is False:
                    break;

                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);
                frame = cv2.resize(frame,(Config.VIDEO_WIDTH,Config.VIDEO_HEIGHT));

                #Augmenting using translation
                for trans in translations:
                    M = np.float32([[1,0,trans[0]],[0,1,trans[1]]])
                    dst = cv2.warpAffine(frame,M,(Config.VIDEO_WIDTH,Config.VIDEO_HEIGHT));
                    dst = np.expand_dims(dst,axis =-1);
                    dst = dst/255.0;
                    frames[counter][frameCounter] = dst;
                    counter+=1;

                    
                #------------------------------------------------------------

                #Augmentation using rotation
                for rot in rotations:
                    M = cv2.getRotationMatrix2D(((Config.VIDEO_WIDTH-1)/2.0,(Config.VIDEO_HEIGHT-1)/2.0),rot,1)
                    dst = cv2.warpAffine(frame,M,(Config.VIDEO_WIDTH,Config.VIDEO_HEIGHT))
                    dst = np.expand_dims(dst,axis =-1);
                    dst = dst/255.0;
                    
                    frames[counter][frameCounter] = dst;
                    counter+=1;
                #-------------------------------------------------------------

                #Augmentation using gaussian filter
                for fil in gaussianFilters:
                    dst = cv2.GaussianBlur(frame,(5,5),fil);
                    dst = np.expand_dims(dst,axis =-1);
                    dst = dst/255.0;
                    frames[counter][frameCounter] = dst;
                    counter+=1;
                    
                #-------------------------------------------------------------

                #Augmentation by changing brightness
                for b in brightness:
                    dst = frame;
                    dst = dst.astype(np.float);
                    dst = dst + dst*(b/100);
                    dst = dst.astype(np.uint8);
                    dst = np.expand_dims(dst,axis =-1);
                    dst = dst/255.0;
                    frames[counter][frameCounter] = dst;
                    counter+=1;
                    # cv2.imshow("f",dst);
                    # cv2.waitKey();
                #-------------------------------------------------------------

                frameCounter+=1;

            pass

            #Save all augmented datas
            augCount = 0;
            for f in frames:
                
                if frameCounter > Config.VIDEO_UNIFORM_LENGTH:
                    f = f[0:frameCounter,:,:,:];
                    randFrames = np.zeros(Config.VIDEO_UNIFORM_LENGTH)

                    S = frameCounter;
                    div = (S/Config.VIDEO_UNIFORM_LENGTH);
                    scale = math.floor(div);

                    #Sample frames using uniform sampling with temporal jitter
                    indices = [];
                    randFrames[::] = div*np.arange(0, Config.VIDEO_UNIFORM_LENGTH) + \
                        float(scale)/2*(np.random.random(size=Config.VIDEO_UNIFORM_LENGTH)-0.5)

                    randFrames[0] = max(randFrames[0], 0)
                    randFrames[Config.VIDEO_UNIFORM_LENGTH-1] = min(randFrames[Config.VIDEO_UNIFORM_LENGTH-1], frameCounter-1)
                    randFrames = np.floor(randFrames);

                    randFrames = randFrames.astype(dtype=np.int32);

                    f = np.take(f,randFrames,axis = 0);   
                #------------------------------------------------------
            
                else:
                    numToPad = Config.VIDEO_UNIFORM_LENGTH -  len(f);
                    f = f[0:Config.VIDEO_UNIFORM_LENGTH,:,:,:];
                    for i in range(frameCounter,Config.VIDEO_UNIFORM_LENGTH):
                        f[i] = f[frameCounter-1];

                #Save frames
                fileFrames = open(os.path.sep.join([Config.BASE_PATH,"frames",str(fileCount+augCount) + ".frames"]) ,'wb');
                fileLabels = open(os.path.sep.join([Config.BASE_PATH,"frames",str(fileCount+augCount) + ".label"]) ,'wb');


                f = tf.convert_to_tensor(tf.expand_dims(f,axis=0));
                lbl = tf.convert_to_tensor(tf.expand_dims(label,axis=0));

                pickle.dump(f,fileFrames);
                pickle.dump(lbl,fileLabels);
            
                fileFrames.close();
                fileLabels.close();

                fileCount+=1;
                augCount+=1;

                print(str((fileCount / totalFiles)*100) + " %");
                print
                #-------------------------------------------------------
            
        #------------------------------------------------------------------
                
    def loadGestures(self,gestures):
        path = os.path.sep.join([Config.BASE_PATH,Config.TRAIN,"frames"]);

        labels_train = open(os.path.sep.join([Config.BASE_PATH,Config.TRAIN,"train_list.txt"]),'r');
        lineCounter = 0;

        for line in labels_train:
            line  = line.rstrip();
            label = line.split(" ")[-1];
            if label in gestures:
                self.trainVideosT.append(os.path.sep.join([path,str(lineCounter) + ".frames"]));
                self.trainLabelsT.append(os.path.sep.join([path,str(lineCounter) + ".label"]));

            lineCounter+=1;
            pass
        
        self.validVideos = self.trainVideosT[int(np.ceil(len(self.trainLabelsT)*0.8)):];
        self.trainVideosT = self.trainVideosT[:int(np.ceil(len(self.trainLabelsT)*0.8))];

        self.validLabels = self.trainLabelsT[int(np.ceil(len(self.trainLabelsT)*0.8)):];
        self.trainLabelsT = self.trainLabelsT[:int(np.ceil(len(self.trainLabelsT)*0.8))];
        
        self.trainDataSize = len(self.trainVideosT);
        self.validDataSize = len(self.validLabels);
        self.loaded = True;

    def load(self):

        #Load train data
        path = os.path.sep.join([Config.BASE_PATH,"frames"]);
        allVideos = glob.glob(path + "/*.frames");
        allLabels = glob.glob(path + "/*.label");

        allVideos,allLabels = shuffle(allVideos,allLabels,random_state=0);


        self.trainVideos = allVideos[:int(np.ceil(len(allVideos)*0.8))];
        self.validVideos = allVideos[int(np.ceil(len(allVideos)*0.8)):];

        self.trainLabels = allLabels[:int(np.ceil(len(allLabels)*0.8))];
        self.validLabels = allLabels[int(np.ceil(len(allLabels)*0.8)):];



        self.trainDataSize = len(self.trainVideos);
        self.validDataSize = len(self.validVideos);



        self.loaded = True;

    def fetchTrain(self, batchSize):
        
        assert self.loaded is True, "Data hasn't loaded yet...";
        if(self.trainDataSize - self.trainIdx < batchSize):
            size = self.trainDataSize - self.trainIdx;
        else:
            size = batchSize;
        retFrames = None;
        retLabels = None;

        # #Load all videos in this batch
        first = True;
        for i in range(size):
            frames = pickle.load(open(self.trainVideos[self.trainIdx],'rb'));
            label = pickle.load(open(self.trainLabels[self.trainIdx],'rb'));

            if first is True:
                retFrames = frames;
                retLabels = label;
                first = False;
            else:
                retFrames = tf.concat([retFrames,frames],axis=0);
                retLabels = tf.concat([retLabels,label],axis=0);

            self.trainIdx += 1;

        if self.trainIdx == self.trainDataSize:
            self.trainIdx = 0;

        return retFrames, retLabels,size;

    def fetchTest(self,batchSize):
        assert self.loaded is True, "Data hasn't loaded yet...";
        retImg = np.zeros(shape=(self.testDataSize, Config.IMG_WIDTH, Config.IMG_HEIGHT, 1), dtype=np.float32);
        retheight = np.zeros(shape=(self.testDataSize, Config.NETWORK_SIZE * Config.NETWORK_SIZE), dtype=np.float32);

        for i in range(self.testDataSize):
            # if self.testIdx == self.testDataSize:
            #     self.testIdx = 0;
            #     break;
            retheight[i] = self.testHeight[i];

            retImg[i] = self.testImage[i];
            #self.testIdx += 1;
        return retImg, retheight;

    def fetchValid(self, batchSize):
        
        assert self.loaded is True, "Data hasn't loaded yet...";
        if(self.validDataSize - self.validIdx < batchSize):
            size = self.validDataSize - self.validIdx;
        else:
            size = batchSize;
        retFrames = None;
        retLabels = None;

        # #Load all videos in this batch
        first = True;
        for i in range(size):
            frames = pickle.load(open(self.validVideos[self.validIdx],'rb'));
            label = pickle.load(open(self.validLabels[self.validIdx],'rb'));

            if first is True:
                retFrames = frames;
                retLabels = label;
                first = False;
            else:
                retFrames = tf.concat([retFrames,frames],axis=0);
                retLabels = tf.concat([retLabels,label],axis=0);

            self.validIdx += 1;

        if self.validIdx == self.validDataSize:
            self.validIdx = 0;

        return retFrames, retLabels,size;

    def getTrainDataSize(self):
        return self.trainDataSize;

    def getTestDataSize(self):
        return self.testDataSize;

    def getTotalDataSize(self):
        return self.totalDataSize;

    def getRandomValidation(self):
        assert self.loaded is True, "Data hasn't loaded yet...";

        r = np.random.randint(0,self.validDataSize);

        retFrames = pickle.load(open(self.validVideos[r],'rb'));
        retLabels = pickle.load(open(self.validLabels[r],'rb'));

        ref = retFrames.numpy();
        retl = retLabels.numpy();
            
        return retFrames, retLabels;

    def getRandomTraining(self):
        retImg = np.zeros(shape=(1, Config.IMG_WIDTH, Config.IMG_HEIGHT, 3), dtype=np.float32);
        retheight = np.zeros(shape=(1, Config.IMG_WIDTH, Config.IMG_HEIGHT, 3
                                    ), dtype=np.float32);

        i = np.random.random_integers(low=0, high=self.trainDataSize - 1);
        retheight[0] = self.trainHeight[i];
        retImg[0] = self.trainImage[i];


        return retImg, retheight;

    def getValidDataSize(self):
        return self.validDataSize;