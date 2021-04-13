import matplotlib
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
import math
import tensorflow as tf
import Fetcher
import Model
import Config
from tqdm.autonotebook import tqdm

fetcher = Fetcher.DataFetcher();
#Uncomment these lines only for the first time to process dataset
#fetcher.processAndSave();
#fetcher.loadGestures(gestures= ["1","2"]);
fetcher.load(); 

numIterationEpochTrain = math.ceil(fetcher.trainDataSize / Config.BATCH_SIZE);
numIterationEpochValid = math.ceil(fetcher.validDataSize / Config.BATCH_SIZE);


model = Model.Net(numBatchTrain = numIterationEpochTrain);
model.network.summary();

startEpoch = model.load_checkpoint();

allLossTrain = [];
allAccTrain = [];

allLossValid = [];
allAccValid = [];


for i in tf.range(start = startEpoch, limit = Config.EPOCHS,dtype=tf.int64):
  print("[INFO]epoch : {}".format(i))
  for step in tqdm(range(numIterationEpochTrain)):
    image, labels ,size = fetcher.fetchTrain(Config.BATCH_SIZE);
  
    loss,out = model.trainStep(input = image,label = labels);

    GT = np.array(labels).argmax(axis=1);
    Pred = out.numpy().argmax(axis=1);
    acc = 0;
    for a in range(size):
      if(GT[a] == Pred[a]):
        acc+=1;

    acc = acc / size;
    
    with model.summary_writer.as_default():
      tf.summary.scalar("loss", loss.numpy(), step= i*numIterationEpochTrain + step)
      tf.summary.scalar("acc", acc, step= i*numIterationEpochTrain + step)
    model.summary_writer.flush()

    loss = loss.numpy();
    allLossTrain.append(loss);
    allAccTrain.append(acc);

  for step in tf.range(numIterationEpochValid, dtype=tf.int64):
    image, labels, size = fetcher.fetchValid(Config.BATCH_SIZE);
    loss,out = model.validStep(input = image,label = labels);

    GT = GT = np.array(labels).argmax(axis=1);
    Pred = out.numpy().argmax(axis=1);
    acc = 0;
    for a in range(size):
      if(GT[a] == Pred[a]): 
        acc+=1;

    acc = acc / size;

    with model.summary_writer.as_default():
      tf.summary.scalar("lossValid", loss.numpy(), step= i*numIterationEpochValid + step)
      tf.summary.scalar("accValid", acc, step= i*numIterationEpochValid + step)
    model.summary_writer.flush()

    loss = loss.numpy();
    allLossValid.append(loss);
    allAccValid.append(acc);

  print("loss train:{}".format(np.mean(allLossTrain)));
  print("accuracy train:{}".format(np.mean(allAccTrain)));

  print("loss valid:{}".format(np.mean(allLossValid)));
  print("accuracy valid:{}".format(np.mean(allAccValid)));
  print("==================================");

  allLossTrain = [];
  allAccTrain = [];

  allLossValid = [];
  allAccValid = [];

  model.save_checkpoint(step = i);