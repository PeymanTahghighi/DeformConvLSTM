# ==================================================================================
# ==================================================================================
import tensorflow as tf
import os
import time
from matplotlib import pyplot as plt
import datetime
import numpy as np
import Config


from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

from DeformableConvLSTM import *


writer = tf.summary.create_file_writer("logs/")


# ==================================================================================
# ==================================================================================

# ----------------------------------------------------------------------------------
class Net():
    def __init__(self,numBatchTrain):
        # Define learning rate and optimizers
        self.learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=Config.LEARNING_RATE,
            decay_steps=20 * numBatchTrain, decay_rate=0.5, staircase=True);
        self.optimizer = tf.keras.optimizers.RMSprop(self.learning_rate_schedule);
        # ---------------------------------------------------------------------------------------

        # Define generator and discriminator models
        self.embedding = tf.keras.layers.Masking(mask_value=0.0)
        self.network = self.build();
        # ---------------------------------------------------------------------------------------

        # Define checkpoint and logging
        self.logdir = "logs/";
        # print(tf.version.VERSION);
        
        self.summary_writer = tf.summary.create_file_writer(
             self.logdir);
        tf.summary.trace_on(graph=True, profiler=True)
        #self.trace(tf.zeros((4 , Config.VIDEO_UNIFORM_LENGTH , Config.VIDEO_WIDTH,  Config.VIDEO_HEIGHT, 1)))
        with self.summary_writer.as_default():
             tf.summary.trace_export(name="model_trace", step=0, profiler_outdir="loggraph");

        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.optimizer,
                                              generator=self.network)
        self.chekcpoint_manager = tf.train.CheckpointManager(self.checkpoint, 'tf_ckpts', max_to_keep=100
                                                             )
        # ---------------------------------------------------------------------------------------- 

    @tf.function
    def trace(self,x):
        return self.network(x)

    def save_checkpoint(self, step):
        path = self.chekcpoint_manager.save()
        print("Saved checkpoint for step {}: {}".format(int(step), path))
        pass

    def load_checkpoint(self):
        if self.chekcpoint_manager.latest_checkpoint:
            idx = self.chekcpoint_manager.latest_checkpoint.find('-');
            epoch = (int(self.chekcpoint_manager.latest_checkpoint[idx + 1:]));

            self.checkpoint.restore(self.chekcpoint_manager.latest_checkpoint)
            print('[INFO]Restored from epoch : {}'.format(int(epoch + 1)));
            return int(epoch + 1);
        return 0;

    def build(self):

        initializer = tf.random_normal_initializer(0., 0.02)
        inp = tf.keras.layers.Input(shape=[Config.VIDEO_UNIFORM_LENGTH, Config.VIDEO_WIDTH,Config.VIDEO_HEIGHT,1], name='input_video');
        
        x = self.conv3d(kernel_size= (3,3,3),filters=64,stride = (1,1,1),initializer = initializer)(inp);
        x = self.avgPool3D(pool_size = (1,2,2),stride = (1,2,2))(x);

        x = self.conv3d(kernel_size= (3,3,3),filters=128,stride = (1,1,1),initializer = initializer,apply_batchnorm=True)(x);
        x = self.avgPool3D(pool_size = (2,2,2),stride = (2,2,2))(x);

        x = self.conv3d(kernel_size= (3,3,3),filters=256,stride = (1,1,1),initializer = initializer,apply_batchnorm=True)(x);
 

        x = self.deformableConvLSTM2D(filters = 32,kernel_size=(3,3),initializer = initializer,return_sequence=True,stride = (1,1),
        bidirectional = False,apply_batchnorm=True,frames=[0,10,20])(x);

        x = self.deformableConvLSTM2D(filters = 32,kernel_size=(3,3),initializer = initializer,return_sequence=True,stride = (1,1),
        bidirectional = False,apply_batchnorm=True,frames=[1,11,21])(x);

        x = self.deformableConvLSTM2D(filters = 32,kernel_size=(3,3),initializer = initializer,return_sequence=True,stride = (1,1),
        bidirectional = False,apply_batchnorm=True,frames=[2,12,22])(x);


        x = self.conv3d(kernel_size= (1,3,3),filters=128,stride = (1,1,1),initializer = initializer,apply_batchnorm=True)(x);

        x = self.avgPool3D(pool_size = (2,2,1),stride = (1,2,2))(x);

        x = self.conv3d(kernel_size= (1,3,3),filters=256,stride = (1,1,1),initializer = initializer,apply_batchnorm=True)(x);

        x = self.avgPool3D(pool_size = (1,2,2),stride = (1,2,2))(x);

        x = self.conv3d(kernel_size= (1,3,3),filters=256,stride = (1,1,1),initializer = initializer,apply_batchnorm=True)(x);

        x = self.avgPool3D(pool_size = (1,2,2),stride = (1,2,2))(x);
        
        x = tf.keras.layers.GlobalAveragePooling3D()(x)

        x = tf.keras.layers.Flatten()(x);
        out = tf.keras.layers.Dense(17,activation='softmax')(x);
 
        return Model(inputs = inp,outputs = out);

        pass
    def conv2d(self, filters, kernel_size, stride, initializer, apply_batchnorm=False
               , activation_func=tf.keras.layers.LeakyReLU(0.2)):
        result = tf.keras.Sequential();
        result.add(tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same',
                                          kernel_initializer=initializer, use_bias=False));
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(activation_func)

        return result

    def conv3d(self, filters, kernel_size, stride, initializer, apply_batchnorm=False
               , activation_func=tf.keras.layers.LeakyReLU(0.2)):
        result = tf.keras.Sequential();
        result.add(tf.keras.layers.Conv3D(filters, kernel_size, strides=stride, padding='same',
                                          kernel_initializer=initializer, use_bias=False,data_format = "channels_last"));
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(activation_func)

        return result

    def deformableConv3d(self, filters, kernel_size, stride, initializer, apply_batchnorm=False
               , activation_func=tf.keras.layers.LeakyReLU(0.2)):
        result = tf.keras.Sequential();
        result.add(DeformableConvLayer3D(filters, kernel_size, strides=stride, padding='same',
                                          kernel_initializer=initializer, use_bias=False,data_format = "channels_last"));
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(activation_func)

        return result


    def deformableConvLSTM2D(self,filters,kernel_size,stride,initializer,return_sequence,bidirectional,frames,apply_batchnorm = False,activation_func=tf.keras.layers.LeakyReLU(0.2)):
        result = tf.keras.Sequential();
        result.add(tf.keras.layers.RNN(DeformableConvLSTMCell(shape = [28,28],kernel = [3,3],
        depth = filters,initializer = initializer,frames = frames),return_sequences = True))
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(activation_func)
        return result;

    def maxPool2D(self, pool_size, stride=2):
        return tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=stride, padding='same');

    def avgPool3D(self, pool_size, stride=2,padding = 'same'):
        return tf.keras.layers.AveragePooling3D(pool_size=pool_size, strides=stride, padding=padding,data_format = "channels_last");

    def fullyConnected(self, out, initializer, activation_func=tf.keras.layers.Activation('relu'),
                       use_bias=False, use_activation=True, apply_dropout=True):
        result = tf.keras.Sequential();
        result.add(tf.keras.layers.Dense(units=out,
                                         kernel_initializer=initializer, use_bias=use_bias));

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(Config.DROPOUT_RATIO))

        if (use_activation):
            result.add(activation_func)

        return result

    def lossFunction(self,GT,generated):
        lossObject = tf.keras.losses.CategoricalCrossentropy(from_logits = True);
        return lossObject(GT,generated);

    @tf.function
    def trainStep(self,input,label):
        with tf.GradientTape() as tape:

            gen_output = self.network(input,training=True);

            loss = self.lossFunction(label,gen_output);

        gradients = tape.gradient(loss,self.network.trainable_variables);

        self.optimizer.apply_gradients(zip(gradients,self.network.trainable_variables));

        return loss,gen_output;
        
    @tf.function
    def validStep(self,input,label):
        gen_output = self.network(input,training=False);

        loss = self.lossFunction(label,gen_output);

        return loss,gen_output;
