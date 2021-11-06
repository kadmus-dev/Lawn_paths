import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Activation, Multiply, Add, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, \
    BatchNormalization, concatenate, Concatenate
from tensorflow.keras.models import Model

""" Different metrics """


@tf.autograph.experimental.do_not_convert
def dice(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


@tf.autograph.experimental.do_not_convert
def iou(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


@tf.autograph.experimental.do_not_convert
def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true + y_pred)
    jac = (intersection + 1.) / (union - intersection + 1.)
    return K.mean(jac)


@tf.autograph.experimental.do_not_convert
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


@tf.autograph.experimental.do_not_convert
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


@tf.autograph.experimental.do_not_convert
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


""" Loss function """


@tf.autograph.experimental.do_not_convert
def weighted_binary_crossentropy(y_true, y_pred):
    zero_weight = 1
    one_weight = 10
    b_ce = K.binary_crossentropy(y_true, y_pred)
    weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
    weighted_b_ce = weight_vector * b_ce
    return K.mean(weighted_b_ce)


""" Nested Attention U-Net architecture """


# g = upsample, x = block_i_out, inter_channel = down_layer_channels // 4
def attention_block(x, g, inter_channel):
    theta_x = Conv2D(inter_channel, (1, 1), strides=(1, 1), padding="same")(x)
    phi_g = Conv2D(inter_channel, (1, 1), strides=(1, 1), padding='same')(g)
    f = Activation("relu")(Add()([theta_x, phi_g]))
    psi_f = Conv2D(1, (1, 1), strides=(1, 1), padding="same")(f)
    rate = Activation("sigmoid")(psi_f)
    att_x = Multiply()([x, rate])
    return att_x


def cba_block(inputs, filters):
    x = Conv2D(filters, (3, 3), kernel_initializer="he_uniform", padding="same")(inputs)
    x = BatchNormalization(fused=True, dtype=tf.float32)(x)
    return Activation("relu")(x)


def AttentionUnet(img_size, wide=False):
    input_shape = (img_size, img_size, 3)
    inputs = Input(input_shape, name="input")

    if wide:
        img_size *= 2

    ker = "he_uniform"

    # Block 1
    filters = img_size // 16
    x = cba_block(inputs, filters)
    block_1_out = cba_block(x, filters)
    pool1 = MaxPooling2D(pool_size=(2, 2), name="block12_pool")(block_1_out)

    # Block 2
    filters = img_size // 8
    x = cba_block(pool1, filters)
    block_2_out = cba_block(x, filters)
    pool2 = MaxPooling2D(pool_size=(2, 2), name="block23_pool")(block_2_out)

    # Block 3
    filters = img_size // 4
    x = cba_block(pool2, filters)
    block_3_out = cba_block(x, filters)
    pool3 = MaxPooling2D(pool_size=(2, 2), name="block34_pool")(block_3_out)

    # Block 4
    filters = img_size // 2
    x = cba_block(pool3, filters)
    x = cba_block(x, filters)
    block_4_out = Dropout(0.4)(x)
    pool4 = MaxPooling2D(pool_size=(2, 2), name="block45_pool")(block_4_out)

    # Block 5
    filters = img_size
    x = cba_block(pool4, filters)
    x = cba_block(x, filters)
    x = cba_block(x, filters)
    block_5_out = Dropout(0.1)(x)

    # UP1
    filters = img_size // 2
    x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same", activation="relu", name="upsample1")(
        block_5_out)
    att1 = attention_block(block_4_out, x, filters // 2)
    x = concatenate([x, att1], name="up1_concatenate")
    x = cba_block(x, 2 * filters)
    x = cba_block(x, 2 * filters)

    # UP2
    filters = img_size // 4
    x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same", activation="relu", name="upsample2")(x)
    att2 = attention_block(block_3_out, x, filters // 2)
    x = concatenate([x, att2], name="up2_concatenate")
    x = cba_block(x, 2 * filters)
    x = cba_block(x, 2 * filters)

    # UP3
    filters = img_size // 8
    x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same", activation="relu", name="upsample3")(x)
    att3 = attention_block(block_2_out, x, filters // 2)
    x = concatenate([x, att3], name="up3_concatenate")
    x = cba_block(x, 2 * filters)
    x = cba_block(x, 2 * filters)

    # UP4
    filters = img_size // 16
    x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same", activation="relu", name="upsample4")(x)
    att4 = attention_block(block_1_out, x, filters // 2)
    x = concatenate([x, att4], name="up4_concatenate")
    x = cba_block(x, 2 * filters)
    x = cba_block(x, 2 * filters)

    outputs = Conv2D(1, (1, 1), activation="sigmoid", kernel_initializer=ker, padding="same", name="output")(x)

    model = Model(inputs=[inputs], outputs=[outputs], name="Attention-Unet-" + str(img_size))

    return model


""" U-Net architecture with pre-trained VGG19 as encoder part """


def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def vgg19_unet(image_size):
    """ Input """
    input_shape = (image_size, image_size, 3)
    inputs = Input(input_shape)

    """ Pre-trained VGG19 Model """
    vgg19 = tf.keras.applications.VGG19(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = vgg19.get_layer("block1_conv2").output  ## (512 x 512)
    s2 = vgg19.get_layer("block2_conv2").output  ## (256 x 256)
    s3 = vgg19.get_layer("block3_conv4").output  ## (128 x 128)
    s4 = vgg19.get_layer("block4_conv4").output  ## (64 x 64)

    """ Bridge """
    b1 = vgg19.get_layer("block5_conv4").output  ## (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)  ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)  ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)  ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)  ## (512 x 512)

    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="VGG19_U-Net")
    return model
