# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf
from DeformableConv2D import *
import Config

class DeformableConvLSTMCell(tf.keras.layers.Layer):
    '''Convolutional LSTM (Long short-term memory unit) recurrent network cell.

    The class uses optional peep-hole connections, optional cell-clipping,
    optional normalization layer, and an optional recurrent dropout layer.

    Basic implmentation is based on tensorflow, tf.nn.rnn_cell.LSTMCell.

    Default LSTM Network implementation is based on:

        http://www.bioinf.jku.at/publications/older/2604.pdf

    Sepp Hochreiter, Jurgen Schmidhuber.
    "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.

    Peephole connection is based on:

        https://research.google.com/pubs/archive/43905.pdf

    Hasim Sak, Andrew Senior, and Francoise Beaufays.
    "Long short-term memory recurrent neural network architectures for large scale acoustic modeling". 2014.

    Default Convolutional LSTM implementation is based on:

        https://arxiv.org/abs/1506.04214

    Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-kin Wong, Wang-chun Woo.
    "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting". 2015.

    Recurrent dropout is base on:
    
        https://arxiv.org/pdf/1603.05118

    Stanislau Semeniuta, Aliaksei Severyn, Erhardt Barth
    "Recurrent Dropout without Memory Loss". 2016.

    Normalization layer is applied prior to nonlinearities.
    '''
    def __init__(self,
                 shape,
                 kernel,
                 depth,
                 frames,
                 strides = (1,1),
                 dilation_rate = (1, 1),
                 padding = "same",
                 use_peepholes=False,
                 cell_clip=None,
                 initializer=None,
                 forget_bias=1.0,
                 activation=None,
                 normalize=None,
                 dropout=None,
                 reuse=None,
                 dynamic=True,
                  **kwargs
                  ):
        '''Initialize the parameters for a ConvLSTM Cell.

        Args:
            shape: list of 2 integers, specifying the height and width 
                of the input tensor.
            kernel: list of 2 integers, specifying the height and width 
                of the convolutional window.
            depth: Integer, the dimensionality of the output space.
            use_peepholes: Boolean, set True to enable diagonal/peephole connections.
            cell_clip: Float, if provided the cell state is clipped by this value 
                prior to the cell output activation.
            initializer: The initializer to use for the weights.
            forget_bias: Biases of the forget gate are initialized by default to 1
                in order to reduce the scale of forgetting at the beginning of the training.
            activation: Activation function of the inner states. Default: `tanh`.
            normalize: Normalize function, if provided inner states is normalizeed 
                by this function.
            dropout: Float, if provided dropout is applied to inner states 
                with keep probability in this value.
            reuse: Boolean, whether to reuse variables in an existing scope.
        '''
        super(DeformableConvLSTMCell, self).__init__( **kwargs)

        tf_shapeHidden = tf.TensorShape(shape + [depth])
        tf_shapeState = tf.TensorShape(shape + [depth])
        self._output_size = tf_shapeHidden;
        self._state_size = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(tf_shapeState, tf_shapeHidden)

        self.filters = shape;
        self._kernel = kernel
        self._depth = depth
        self.strides = strides;
        self.padding = padding;
        self.dilation_rate = dilation_rate;
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializer
        self.gateBiasInitializer = tf.random_normal_initializer(0.0, 0.0);
        self.inputBiasInitializer = tf.random_normal_initializer(0.0, 0.0);
        self.forgetBiasInitializer = tf.random_normal_initializer(mean = 1.0, stddev = 0.0);
        self.outputBiasInitializer = tf.random_normal_initializer(0.0, 0.0);

        
        self._activation = activation or tf.nn.tanh
        self._normalize = normalize
        self._dropout = dropout

        self.gateBias = None;
        self.inputBias = None;
        self.forgetBias = None;
        self.outputBias = None;

        self.first = True;

        self.wxi = None;
        self.whi = None;

        self.wxf = None;
        self.whf = None;

        self.wxo = None;
        self.who = None;

        self.num_deformable_group = depth;

        self.counter = 0;
        self.deformableInterval = 100;
        self.frames = frames;

        self.wconvInput = None;
        self.wconvHidden = None;
        if self._use_peepholes:
            self._w_f_diag = None
            self._w_i_diag = None
            self._w_o_diag = None

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, states):
        '''Run one step of ConvLSTM.

        Args:
            inputs: input Tensor, 4D, (batch, shape[0], shape[1], depth)
            state: tuple of state Tensor, both `4-D`, with tensor shape `c_state` and `m_state`.

        Returns:
            A tuple containing:

            - A '4-D, (batch, height, width, depth)', Tensor representing 
                the output of the ConvLSTM after reading `inputs` when previous 
                state was `state`.
                Here height, width is:
                    shape[0] and shape[1].
            - Tensor(s) representing the new state of ConvLSTM after reading `inputs` when
                the previous state was `state`. Same type and shape(s) as `state`.
        '''
        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(4)[3]
        if input_size is None:
            raise ValueError('Could not infer size from inputs.get_shape()[-1]')

        state_prev, hidden_prev = states;

        #inputs = tf.concat([inputs, hidden_prev], axis=-1)

        if self.first is True:
            kernel_shape = self._kernel + [inputs.shape[-1], self._depth]
            kernel_shape = tf.TensorShape(kernel_shape);
            self.wconvInput = tf.Variable(lambda : self._initializer(shape=kernel_shape,dtype=tf.float32), dtype = dtype,name = "wconvInput");

            kernel_shape = [inputs.shape[1], inputs.shape[2],self._depth]
            kernel_shape = tf.TensorShape(kernel_shape);
            self.gateBias =  tf.Variable(lambda : self.gateBiasInitializer(shape=kernel_shape, dtype=tf.float32), dtype = dtype,name = "inputBias");

            kernel_shape = self._kernel + [hidden_prev.shape[-1], self._depth]
            kernel_shape = tf.TensorShape(kernel_shape);
            self.wconvHidden = tf.Variable(lambda : self._initializer(shape=kernel_shape, dtype=tf.float32), dtype = dtype,name = "wconvHidden");

            kernel_shape = [self._depth]
            kernel_shape = tf.TensorShape(kernel_shape);
            self.wxi = tf.Variable(lambda : self._initializer(shape=kernel_shape, dtype=tf.float32), dtype = dtype,name = "wxi");
            self.whi = tf.Variable(lambda : self._initializer(shape=kernel_shape, dtype=tf.float32), dtype = dtype,name = "whi");
            self.inputBias = tf.Variable(lambda : self.inputBiasInitializer(shape=kernel_shape, dtype=tf.float32), dtype = dtype,name = "inputBias");

            self.wxf = tf.Variable(lambda : self._initializer(shape=kernel_shape, dtype=tf.float32), dtype = dtype,name = "wxf");
            self.whf = tf.Variable(lambda : self._initializer(shape=kernel_shape, dtype=tf.float32), dtype = dtype,name = "whf");
            self.forgetBias = tf.Variable(lambda : self.forgetBiasInitializer(shape=kernel_shape, dtype=tf.float32), dtype = dtype,name = "forgetBias");

            self.wxo = tf.Variable(lambda : self._initializer(shape=kernel_shape, dtype=tf.float32), dtype = dtype,name = "wxo");
            self.who = tf.Variable(lambda : self._initializer(shape=kernel_shape, dtype=tf.float32), dtype = dtype,name = "who");
            self.outputBias = tf.Variable(lambda : self.outputBiasInitializer(shape=kernel_shape, dtype=tf.float32), dtype = dtype,name = "outputBias");

            # input_dim = int(inputs.get_shape()[-1])

            # offset_num = self._kernel[0] * self._kernel[1] * self.num_deformable_group;
            # self.offset_layer_kernel = self.add_weight(
            #     name='offset_layer_kernel',
            #     shape=self._kernel + [input_dim,offset_num*2],  # 2 means x and y axis
            #     initializer=tf.zeros_initializer(),
            #     regularizer=None,
            #     trainable=True,
            #     dtype=tf.float32);

            # self.offset_layer_bias = self.add_weight(
            #     name='offset_layer_bias',
            #     shape=(offset_num * 2,),
            #     initializer=tf.zeros_initializer(),
            #     # initializer=tf.random_uniform_initializer(-5, 5),
            #     regularizer=None,
            #     trainable=True,
            #     dtype=tf.float32);
           # w = tf.Variable(lambda : self._initializer(shape=kernel_shape, dtype=tf.float32))

        # i = input_gate, j = new_input, f = forget_gate, o = ouput_gate

        #concat = tf.concat([inputs, m_prev], axis=-1);

        # offset = tf.nn.conv2d(inputs,
        #                       filters=self.offset_layer_kernel,
        #                       strides=[1, *self.strides, 1],
        #                       padding=self.padding.upper(),
        #                       dilations=[1, *self.dilation_rate, 1])
        # offset += self.offset_layer_bias; 
        
        if self.counter%16 in self.frames :
            tf.print(self.counter);
            outInputs = DeformableConvLayer2D(filters = self._depth,kernel_size = (self._kernel[0],self._kernel[1]),kernel_initializer=self._initializer)(inputs);
        else:
            outInputs = tf.nn.conv2d(inputs, self.wconvInput, (1, 1, 1, 1), 'SAME')

        g = tf.nn.tanh(outInputs +
        tf.nn.conv2d(hidden_prev, self.wconvHidden, (1, 1, 1, 1), 'SAME') + self.gateBias);

        
        gapG = tf.keras.layers.GlobalAveragePooling2D()(g);
        
        gapInputs = tf.keras.layers.GlobalAveragePooling2D()(outInputs);
       # gapInputs = tf.squeeze(gapInputs);

        gapHidden = tf.keras.layers.GlobalAveragePooling2D()(hidden_prev);
        #gapHidden = tf.squeeze(gapHidden);

        i = tf.nn.sigmoid(tf.multiply(self.wxi,gapInputs) + tf.multiply(self.whi,gapHidden) + self.inputBias);
        f = tf.nn.sigmoid(tf.multiply(self.wxf,gapInputs) + tf.multiply(self.whf,gapHidden) + self.forgetBias);
        o = tf.nn.sigmoid(tf.multiply(self.wxo,gapInputs) + tf.multiply(self.who,gapHidden) + self.outputBias);

        i = tf.reshape(tf.tile(i,[28,28]),[Config.BATCH_SIZE,28,28,i.shape[-1]]);
        f = tf.reshape(tf.tile(f,[28,28]),[Config.BATCH_SIZE,28,28,f.shape[-1]]);
        o = tf.reshape(tf.tile(o,[28,28]),[Config.BATCH_SIZE,28,28,o.shape[-1]]);

        
        #i = tf.stack([i,i,i,i],axis = 0);
        #o = tf.stack([o,o,o,o],axis = 0);
        #f = tf.stack([f,f,f,f],axis = 0);
        state = f * state_prev + i *g;

        hidden = o * tf.nn.tanh(state);

        #state = tf.keras.layers.GlobalAveragePooling2D()(state);

        # # Diagonal connections
        # if self._use_peepholes and not self._w_f_diag:
        #     scope = tf.get_variable_scope()
        #     with tf.variable_scope(scope, initializer=self._initializer):
        #         self._w_f_diag = tf.get_variable('w_f_diag', c_prev.shape[1:], dtype=dtype)
        #         self._w_i_diag = tf.get_variable('w_i_diag', c_prev.shape[1:], dtype=dtype)
        #         self._w_o_diag = tf.get_variable('w_o_diag', c_prev.shape[1:], dtype=dtype)

        # if self._use_peepholes:
        #     f = f + self._w_f_diag * c_prev
        #     i = i + self._w_i_diag * c_prev
        # if self._normalize is not None:
        #     f = self._normalize(f)
        #     i = self._normalize(i)
        #     j = self._normalize(j)

        # j = self._activation(j)

        # if self._dropout is not None:
        #     j = tf.nn.dropout(j, self._dropout)

        # state = tf.nn.sigmoid(f + self._forget_bias) * c_prev + tf.nn.sigmoid(i) * j

        # if self._cell_clip is not None:
        #     # pylint: disable=invalid-unary-operand-type
        #     c = tf.clip_by_value(c, -self._cell_clip, self._cell_clip)
        #     # pylint: enable=invalid-unary-operand-type
        # if self._use_peepholes:
        #     o = o + self._w_o_diag * c
        # if self._normalize is not None:
        #     o = self._normalize(o)
        #     c = self._normalize(c)

        # hidden = tf.nn.sigmoid(o) * self._activation(c)

        new_state = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(state, hidden)
        self.first = False;
        if(inputs.get_shape()[0] is not None):
            
            self.counter+=1;
        return hidden, new_state;