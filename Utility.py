import os
from imutils import paths
import Config
from moviepy.editor import VideoFileClip
import math
import cv2
import numpy as np
import pickle
import Fetcher
import tensorflow as tf

a = tf.reshape( tf.constant( range( 0, 49152 ) ), ( 4, 16,16,16,3 ) )
print(a.numpy())
b = tf.pad( a, [ [ 0, 0], [ 0, 1 ],[ 0, 1 ],[0,1],[0,0] ], "SYMMETRIC" )
print(b.numpy());


def convolve2D(image,kernelSize,filters):
    initializer = tf.random_normal_initializer(0., 0.0)
    n = 4

    indices = [[[[x * n + y ] for y in range(0,n)] for x in range(0,n)]]
    indices = tf.convert_to_tensor(indices);

    image = tf.expand_dims(image,axis=0);

    patches = tf.image.extract_patches(image,sizes = (1,kernelSize[0],kernelSize[1],1),strides=(1,1,1,1),padding = "SAME",rates = (1,1,1,1));
    #print(patches.numpy());
    indices = tf.reshape(indices,[1,n*n]);

    tmpImage = tf.reshape(image,[n*n]);

    output = tf.Variable(lambda : initializer(shape=[patches.get_shape()[1],patches.get_shape()[2],filters], dtype=tf.float32), dtype = tf.float32,name = "output");
    #output = tf.reshape(output,[n*n]);

    initializer = tf.random_normal_initializer(0.0,1.0);
    kernel = tf.Variable(lambda : initializer(shape=[filters,kernelSize[0],kernelSize[1],2], dtype=tf.float32), dtype = tf.float32,name = "kernel");

    for f in range(filters):
        #tmpOutput = tf.Variable(lambda : initializer(shape=[patches.get_shape()[1],patches.get_shape()[2],1], dtype=tf.float32), dtype = tf.float32);
        for i in range(patches.get_shape()[1]):
            for j in range(patches.get_shape()[2]):
                #kernel = tf.expand_dims(kernel,axis=0);
                patch = tf.gather(tmpImage,patches[0][i][j]);
                tmpKernel = tf.reshape(kernel[f],[9]);
                convolved = patch * tmpKernel;

                patch = tf.cast(patch,dtype=tf.int32);
                updateIndices = tf.expand_dims(patches[0][i][j],axis=1);
                #print(updateIndices.numpy());

                #output = tf.compat.v1.scatter_update(output,indices = [i,j],updates = tf.reduce_sum(convolved));
                output[i,j,f].assign(tf.reduce_sum(convolved));
        #output = tf.stack([output,tmpOutput],axis = 2);

    print(output.numpy());



#data = Fetcher.DataFetcher();
#data.processAndSave();


n=4;

initializer = tf.random_normal_initializer(1., 1.0)
image = tf.Variable(lambda : initializer(shape=[2,n,n,1], dtype=tf.float32), dtype = tf.float32,name = "output");
indices = tf.Variable(lambda : initializer(shape=[2,n,n,9,64,3], dtype=tf.float32), dtype = tf.float32,name = "output");
indices = tf.cast(indices,dtype=tf.int32);
s = indices.shape[:-1] + image.shape[indices.shape[-1]:]
a = tf.gather_nd(image,indices);

image = tf.squeeze(image);
#print(image.numpy())
#paddedImage = tf.pad( image, [ [ 0, 1 ], [ 0, 1 ]], "SYMMETRIC" )
#print(paddedImage.numpy())
#paddedImage = tf.expand_dims(paddedImage,axis = 2);


# convolve2D(image,[3,3],filters = 64);

feat_h, feat_w = [int(i) for i in [n,n]]


x, y,z = tf.meshgrid(tf.range(10), tf.range(10),tf.range(5));
x = tf.transpose(x,[2,0,1]);
y = tf.transpose(y,[2,0,1]);
z = tf.transpose(z,[2,0,1]);
print(z.numpy())
# _, _,z = tf.meshgrid(tf.range(1), tf.range(1),tf.range(64))

x, y,z = [tf.reshape(i, [1, *i.get_shape(), 1]) for i in [x, y,z]]  # shape [1, h, w, 1]
# _, _,_,z = tf.meshgrid(tf.range(1), tf.range(1),tf.range(9),tf.range(64))

x, y,z = [tf.extract_volume_patches(i,
        [1, 3,3,3, 1],
        [1, 1,1,1, 1],
        'SAME')
for i in [x, y,z]]
print(x.numpy());
#x= tf.expand_dims(x,axis=-1);
#y= tf.expand_dims(y,axis=-1);
pix = tf.stack([x,z],axis = -1);
print(pix.numpy());

xOffset = tf.Variable(lambda : initializer(shape=[9], dtype=tf.float32), dtype = tf.float32,name = "xOffset");
yOffset = tf.Variable(lambda : initializer(shape=[9], dtype=tf.float32), dtype = tf.float32,name = "yOffset");
x = tf.cast(x,tf.float32);
y = tf.cast(y,tf.float32);

#print(x.numpy())
x = x + xOffset;
y = y + yOffset;



x0 = tf.floor(x);
x1 = x0+1;
y0 = tf.floor(y);
y1=y0+1;

x0 = tf.cast(tf.clip_by_value(x0,clip_value_min = 0,clip_value_max = n),dtype=tf.int32);
x1 = tf.cast(tf.clip_by_value(x1,clip_value_min = 0,clip_value_max = n),dtype=tf.int32);
y0 = tf.cast(tf.clip_by_value(y0,clip_value_min = 0,clip_value_max = n),dtype=tf.int32);
y1 = tf.cast(tf.clip_by_value(y1,clip_value_min = 0,clip_value_max = n),dtype=tf.int32);

pixel_idx = tf.stack([x0, y0], axis=-1)
p0 = tf.gather_nd(paddedImage,indices = pixel_idx);

pixel_idx = tf.stack([x1, y0], axis=-1)
p1 = tf.gather_nd(paddedImage,indices = pixel_idx);

pixel_idx = tf.stack([x0, y1], axis=-1)
p2 = tf.gather_nd(paddedImage,indices = pixel_idx);

pixel_idx = tf.stack([x1, y1], axis=-1)
p3 = tf.gather_nd(paddedImage,indices = pixel_idx);

#print("--");
print(x0.numpy())

x1 = tf.cast(x1,tf.float32);
x0 = tf.cast(x0,tf.float32);
y1 = tf.cast(y1,tf.float32);
y0 = tf.cast(y0,tf.float32);

w0 = (y1 - y) * (x1 - x)
w1 = (y1 - y) * (x - x0)
w2 = (y - y0) * (x1 - x)
w3 = (y - y0) * (x - x0)

w0, w1, w2, w3 = [tf.expand_dims(i, axis=-1) for i in [w0, w1, w2, w3]]

pixels = tf.add_n([w0 * p0, w1 * p1, w2 * p2, w3 * p3])


print(pixels.numpy());

# b = tf.Variable(lambda : initializer(shape=[12,12,5], dtype=tf.float32), dtype = tf.float32,name = "b");
# #a = tf.tile(b,[28*28]);
# #a = tf.reshape(a,[28,28,256])
# c= a*b;
# print(a.numpy())


# path = os.path.sep.join([Config.BASE_PATH,Config.TRAIN,"train"]);
# videos_train_paths = list(paths.list_files(path));
# labels_train = open(os.path.sep.join([Config.BASE_PATH,Config.TRAIN,"train_list.txt"]),'r');
# allFrames = [];
# flow = np.zeros((240,320),dtype = np.uint8);
# i = 1;
# for line in labels_train:
#     splited = line.split();
#     frames = [];
#     videoName = splited[0];
#     clip = cv2.VideoCapture(os.path.sep.join([Config.BASE_PATH,Config.TRAIN,videoName]));
#     (grabbed,prevFrame) = clip.read();
#     prevFrame = cv2.cvtColor(prevFrame,cv2.COLOR_BGR2GRAY);
#     frames.append(np.expand_dims(cv2.resize(prevFrame,(64,64)),axis=2))
    
#     while True:
#         (grabbed,frame) = clip.read();

#         if grabbed is False:
#             break;

#         frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);
       
#         flow = cv2.calcOpticalFlowFarneback(prevFrame, frame, None, 0.5,5, 15, 3, 5, 1.2, 0)
#         magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

#         if(np.mean(magnitude) > 0.4):
#             frames.append(np.expand_dims(cv2.resize(frame,(64,64)),axis=2))
#             cv2.imshow("frame",frame);
#             cv2.imshow("mag",magnitude);
#             cv2.waitKey();


#         prevFrame = frame;
#         pass
#     i+=1;
#     allFrames.append(frames);
    

#     # if i == 8:
#     #     allFrames = np.array(allFrames);
#     #     f = open("1-16.dmp",'wb');
#     #     pickle.dump(allFrames,f);
#     #     f.close();
#     # print(i);
#     # print
    
# print(math.ceil(clip.duration * clip.fps));