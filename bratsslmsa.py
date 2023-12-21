#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.layers import Conv3D, BatchNormalization, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Activation, Dense
from tensorflow.keras.layers import MaxPooling3D, UpSampling3D, Reshape, Concatenate, Add
from tensorflow.keras.layers import Conv3DTranspose, Input

def stem_block(input_tensor):
    # First convolution layer with kernel (3*3*3) and stride=2
    conv3_2 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(input_tensor)
    bn1 = tf.keras.layers.BatchNormalization()(conv3_2)
    relu1 = tf.keras.layers.ReLU()(bn1)

    # Second convolution layer with kernel (3*3*3) and stride=1
    conv3_1 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(relu1)
    bn2 = tf.keras.layers.BatchNormalization()(conv3_1)
    relu2 = tf.keras.layers.ReLU()(bn2)

    # Third convolution layer with kernel (3*3*3) and stride=2
    conv3_2 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(relu2)
    bn3 = tf.keras.layers.BatchNormalization()(conv3_2)
    relu3 = tf.keras.layers.ReLU()(bn3)

    # Fourth convolution layer with kernel (3*3*3) and stride=1
    conv3_1 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(relu3)
    bn4 = tf.keras.layers.BatchNormalization()(conv3_1)
    relu4 = tf.keras.layers.ReLU()(bn4)

    return relu4

def Conv3Dcust(input_tensor, filters, kernel_size, strides, padding):

    # First convolution layer with kernel (3*3*3) and stride=1
    conv3_1 = tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=(1, 1, 1), padding='same')(input_tensor)
    bn2 = tf.keras.layers.BatchNormalization()(conv3_1)
    relu1 = tf.keras.layers.ReLU()(bn2)

    # Second convolution layer with kernel (3*3*3) and stride=1
    conv3_1 = tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=(1, 1, 1), padding='same')(relu1)
    bn2 = tf.keras.layers.BatchNormalization()(conv3_1)

    # Add the input tensor to the output of the second convolution layer
    added = tf.keras.layers.Add()([bn2, input_tensor])
    relu2 = tf.keras.layers.ReLU()(added)

    conv3_1 = tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=(1, 1, 1), padding='same')(relu2)
    bn2 = tf.keras.layers.BatchNormalization()(conv3_1)
    relu2 = tf.keras.layers.ReLU()(bn2)

    # Second convolution layer with kernel (3*3*3) and stride=1
    conv3_1 = tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=(1, 1, 1), padding='same')(relu2)
    bn2 = tf.keras.layers.BatchNormalization()(conv3_1)

    # Add the input tensor to the output of the second convolution layer
    added = tf.keras.layers.Add()([bn2, relu2])
    relu2 = tf.keras.layers.ReLU()(added)

    return relu2

class ShortTermMemoryAttentionBlock(Model):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ShortTermMemoryAttentionBlock, self).__init__(**kwargs)
        self.conv_query = Conv3D(filters, kernel_size, padding='same')
        self.conv_key = Conv3D(filters, kernel_size, padding='same')
        self.conv_value = Conv3D(filters, kernel_size, padding='same')
        self.batch_norm = BatchNormalization()
        self.activation = ReLU()

    def call(self, inputs):
        query = self.activation(self.batch_norm(self.conv_query(inputs)))
        key = self.activation(self.batch_norm(self.conv_key(inputs)))
        value = self.activation(self.batch_norm(self.conv_value(inputs)))
               
        # Implement 3D self-attention calculation here
        # Calculate scaled dot-product attention scores
        attention_logits = tf.matmul(query, key, transpose_b=True)
        attention_scores = tf.nn.softmax(attention_logits, axis=-1)

        # Apply attention scores to the value
        attention_output = tf.matmul(attention_scores, value)

        # Add the original input to the attention output (residual connection)
        output = tf.keras.layers.Add()([inputs, attention_output])

        return output

class MSAAttentionBlock(tf.keras.Model):
    def __init__(self, num_heads, d_model, d_k, d_v):
        super(MSAAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.conv_q = Conv3D(num_heads * d_k, (1, 1, 1), padding='same')
        self.conv_k = Conv3D(num_heads * d_k, (1, 1, 1), padding='same')
        self.conv_v = Conv3D(num_heads * d_v, (1, 1, 1), padding='same')
        self.linear1 = Conv3D(2 * num_heads * d_k, (1, 1, 1), padding='same')
        self.conv_output = Conv3D(num_heads * d_v, (1, 1, 1), padding='same')
        self.norm1 = BatchNormalization(epsilon=1e-6)
        self.norm2 = BatchNormalization(epsilon=1e-6)
        self.activation = Activation('relu')

        # Create conv_to_256_channels layer here
        self.conv_to_256_channels = Conv3D(256, (1, 1, 1), padding='same', activation='relu')

    def call(self, inputs):
        inputs1 = tf.identity(inputs)
        inputs2 = tf.identity(inputs)
        q = self.conv_q(inputs1)
        k = self.conv_k(inputs2)
        v = self.conv_v(inputs2)

        q = tf.reshape(q, [-1, self.num_heads, self.d_k])
        k = tf.reshape(k, [-1, self.num_heads, self.d_k])
        v = tf.reshape(v, [-1, self.num_heads, self.d_v])
        
        attn_weights = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.d_k, tf.float32)), axis=-1)
        attn_output = tf.matmul(attn_weights, v)
        attn_output = tf.reshape(attn_output, [-1, 32, 32, 32, self.num_heads * self.d_v])  # Adjust the output shape

        inter = self.activation(self.linear1(attn_output))
        inter = tf.reshape(inter, [-1, 32, 32, 32, 2 * self.num_heads * self.d_k])  # Adjust the output shape

        # The convolution operation is applied on the inter tensor
        inter = self.conv_output(inter)

        # Use the conv_to_256_channels layer defined in the constructor
        inputs1_converted = self.conv_to_256_channels(inputs1)
              
        output = self.norm1(inputs1_converted + inter)
        return output

class LongTermMemoryAttentionBlock(tf.keras.Model):
    def __init__(self, num_heads, d_model, d_k, d_v):
        super(LongTermMemoryAttentionBlock, self).__init__()
        self.msa_block = MSAAttentionBlock(num_heads, d_model, d_k, d_v)
        self.norm1 = BatchNormalization(epsilon=1e-6)

    def call(self, inputs):
        msa_output = self.msa_block(inputs)

        # Use the conv_to_256_channels layer defined in the MSAAttentionBlock constructor
        inputs_converted = self.msa_block.conv_to_256_channels(inputs)
        inter = self.norm1(msa_output + inputs_converted)
        return inter

def reconstruction_block(input_tensor, initial_filter_count=32):
    model = tf.keras.Sequential()
    # BatchNormalization and ReLU for the first convolution layer
    x = BatchNormalization()(input_tensor)
    x = ReLU()(x)
    # Convolutional layer with kernel (3*3*3) and stride=1
    x = Conv3D(filters=initial_filter_count, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    print("1",x.shape)
    # BatchNormalization and ReLU for the second convolution layer
    x = BatchNormalization()(x)
    x = ReLU()(x)
    print("2",x.shape)
    # Convolutional layer with kernel (3*3*3) and stride=2
    x =Conv3DTranspose(filters=initial_filter_count, kernel_size=(3, 3, 3), strides=(2,2,2), padding='same')(x)

    # BatchNormalization and ReLU for the third convolution layer
    x = BatchNormalization()(x)
    x = ReLU()(x)
    print("3",x.shape)
    # Convolutional layer with kernel (3*3*3) and stride=1
    x = Conv3D(filters=initial_filter_count, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)

    # BatchNormalization and ReLU for the fourth convolution layer
    x = BatchNormalization()(x)
    x = ReLU()(x)
    print("4",x.shape)
    # Convolutional layer with kernel (3*3*3) and stride=2
    x =Conv3DTranspose(filters=initial_filter_count, kernel_size=(3, 3, 3), strides=(2,2,2), padding='same')(x)

    # BatchNormalization and ReLU for the fifth convolution layer
    x = BatchNormalization()(x)
    x = ReLU()(x)
    print("5",x.shape)
    # Convolutional layer with kernel (3*3*3) and stride=1
    x = Conv3D(filters=initial_filter_count, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)

    # BatchNormalization and ReLU for the sixth convolution layer
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # Convolutional layer with kernel (3*3*3) and stride=1
    x = Conv3D(filters=initial_filter_count//2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    print("6",x.shape)
    return x


def slm_sa(input_shape, num_classes):
    stem_block_input = Input(input_shape)  # Assumes you have defined stem_block_input shape
    # Pass the entire batch through the stem block
    stem_block_output = stem_block(stem_block_input)
    
    # Short term memory block1 process
    # input_tensor = Input(shape=stem_block_output.shape[1:])# Use the shape of stem_block_output
    conv_layer1_output = Conv3Dcust(stem_block_output, filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')
    downsampling1_output = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv_layer1_output)
    conv2_output = Conv3Dcust(conv_layer1_output, filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')
    downsampling1_output_adjusted = Conv3D(filters=63, kernel_size=(1, 1, 1))(downsampling1_output)
    conv2_output_adjusted = Conv3D(filters=1, kernel_size=(17, 17, 17))(conv2_output)
    combined_input = Concatenate(axis=-1)([downsampling1_output_adjusted, conv2_output_adjusted])
    
    # Features increased dimension reduced combined input to short_term_block1
    short_term_block1_output = ShortTermMemoryAttentionBlock(filters=64, kernel_size=(3, 3, 3))(combined_input)
    upsample_layer = Conv3DTranspose(filters=32, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding='valid')(short_term_block1_output)
    short_term_block1_output_adjusted = Conv3D(filters=32, kernel_size=(1, 1, 1))(upsample_layer)
    

    # Short term memory block2 process
    conv3_input = Add()([conv2_output, short_term_block1_output_adjusted])  # Element-wise addition
    conv3_output = Conv3Dcust(conv3_input, filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')
    downsampling2_output = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(short_term_block1_output)
    downsampling2_output_adjusted = Conv3D(filters=127, kernel_size=(1, 1, 1))(downsampling2_output)
    conv3_output_adjusted = Conv3D(filters=1, kernel_size=(4, 4, 4), strides=(4, 4, 4))(conv2_output)
    combined_input2 = Concatenate(axis=-1)([downsampling2_output_adjusted, conv3_output_adjusted])
 
    # Features increased dimension reduced combined input2 to short_term_block2
    short_term_block2_output = ShortTermMemoryAttentionBlock(filters=128, kernel_size=(3, 3, 3))(combined_input2)
    upsample_layer = Conv3DTranspose(filters=32, kernel_size=(4, 4, 4), strides=(4, 4, 4), padding='valid')(short_term_block2_output)
    short_term_block2_output_adjusted = Conv3D(filters=32, kernel_size=(1, 1, 1))(upsample_layer)
    
    # Short term memory block3 process
    conv4_input = Add()([conv3_output, short_term_block2_output_adjusted])  # Element-wise addition
    conv4_output = Conv3Dcust(conv4_input, filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')
    downsampling3_output = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(short_term_block2_output)
    downsampling3_output_adjusted = Conv3D(filters=255, kernel_size=(1, 1, 1))(downsampling3_output)
    conv4_output_adjusted = Conv3D(filters=1, kernel_size=(8, 8, 8), strides=(8, 8, 8))(conv4_output)
    combined_input3 = Concatenate(axis=-1)([downsampling3_output_adjusted, conv4_output_adjusted])
   
    # Features increased dimension reduced combined input3 to short_term_block3
    short_term_block3_output = ShortTermMemoryAttentionBlock(filters=256, kernel_size=(3, 3, 3))(combined_input3)
    
    # Define input layers for each short-term block output
    upsample_layer1 = Conv3DTranspose(filters=32, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding='valid')(short_term_block1_output)
    merge_short_term_block1 = Conv3D(filters=32, kernel_size=(1, 1, 1))(upsample_layer1)
    upsample_layer2 = Conv3DTranspose(filters=32, kernel_size=(4, 4, 4), strides=(4, 4, 4), padding='valid')(short_term_block2_output)
    merge_short_term_block2 = Conv3D(filters=32, kernel_size=(1, 1, 1))(upsample_layer2)
    upsample_layer3 = Conv3DTranspose(filters=32, kernel_size=(8, 8, 8), strides=(8, 8, 8), padding='valid')(short_term_block3_output)
    merge_short_term_block3 = Conv3D(filters=32, kernel_size=(1, 1, 1))(upsample_layer3)
    

    # Apply the Concatenate layer to the three tensors
    merged_short_block_output = Concatenate(axis=-1)([merge_short_term_block1, merge_short_term_block2, merge_short_term_block3])
    

    # Define the LongTermMemoryAttentionBlock using the concatenated input
    num_heads = 8  # Number of attention heads
    d_model = 256  # Model dimension
    d_k = d_model // num_heads
    d_v = d_model // num_heads  # Dimensions for keys and values

    # Instantiate the LongTermMemoryAttentionBlock
    long_term_memory_block_output = LongTermMemoryAttentionBlock(num_heads, d_model, d_k, d_v)(merged_short_block_output)
    
    upsample_layer = Conv3DTranspose(filters=32, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid')(long_term_memory_block_output)
    long_term_memory_block_output_adjusted = Conv3D(filters=32, kernel_size=(1, 1, 1))(upsample_layer)
   
    reconstruction_output = reconstruction_block(long_term_memory_block_output_adjusted)
    
     # Add the classification layer
#     classification_output = Dense(num_classes, activation='softmax')(reconstruction_output)
#     classification_output_flattened = tf.reshape(classification_output, [-1])
#     classification_output_argmax = tf.argmax(classification_output_flattened, axis=4)
    
#     print("classification_output_np.unique=",tf.unique(classification_output_flattened))
#     print("classification_output_argmax=",classification_output_argmax)

    # Add a 1x1x1 3D convolution layer with softmax activation
    output_conv = Conv3D(num_classes, kernel_size=(1, 1, 1), activation='softmax', padding='same')(reconstruction_output)
    # Create a new model that includes the classification layer
    final_classification_model = Model(inputs=stem_block_input, outputs=output_conv, name="slm_sa")

    return final_classification_model

if __name__ == "__main__":
    input_shape = (128, 128, 128, 3)
    num_classes = 4  
    model = slm_sa(input_shape, num_classes)
    # Summary of the combined model
    model.summary()
    print(model.input_shape)
    print(model.output_shape)


# In[ ]:




