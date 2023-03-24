#Importing required libraries
#We have attempted to do the image segmentation using the U-NET arcgitecture
#The U-NET architecture uses fully convolutional neural netwok
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from keras.models import Model


# We create a convolutional layer in a function to be used further with the call method
# Here num_filters basically is the number of output features
def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)   # We give the kernel siza as 3 and set padding as "same"
    x = BatchNormalization()(x)     # We give the output of conv layer as input to the Normalization layer
    x = Activation("relu")(x)    # We use RELU Activation function to provide non-linearity
    
    return x


# Building the Encoder Block, we give it the convolutional block and Max Pooling layer
# The MaxPooling layer uses the output of the conv_block as its input.
def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

# Here x act as Skip connections and variable p act as the output variable


# Now building the De-Coder block
def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, 2, strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


# Now building the U-NET layer
def build_unet(input_shape):
    inputs = Input(input_shape)
    
    s1, p1 = encoder_block(inputs, 64)  #U-NET kernel number starts from 64
    s2, p2 = encoder_block(p1, 128)    # After each layer of U-NET the number gets doubled  and each layers gets the output of the previous layer
    s3, p3 = encoder_block(p2, 256)     # Then its tripled
    s4, p4 = encoder_block(p3, 512)     # Then its quadrupled
    
    #print(s1.shape, s2.shape, s3.shape, s4.shape)
    #print(p1.shape, p2.shape, p3.shape, p4.shape)
    
    b1 = conv_block(p4, 1024)
    
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="UNET")
    return model
    
    
if __name__ == "__main__":
    input_shape = (512, 512, 3)
    model = build_unet(input_shape)
    model.summary()
    
    
    

