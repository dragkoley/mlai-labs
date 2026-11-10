from keras import KerasTensor, Model
from keras.src.applications.resnet import ResNet152, preprocess_input
from keras.src.layers import Activation, Add, Conv2D, Input, Rescaling, GlobalAveragePooling2D, \
    Dense, LayerNormalization, Dropout


def resnet_block(x: KerasTensor, filters: int, kernel_size: int, stride: int) -> KerasTensor:
    shortcut: KerasTensor = x

    x: KerasTensor = LayerNormalization()(x)
    x: KerasTensor = Activation("relu")(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut: KerasTensor = Conv2D(filters=filters, kernel_size=1, strides=stride, padding="same", use_bias=False)(
            x)

    x: KerasTensor = Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding="same", use_bias=False)(x)

    x: KerasTensor = LayerNormalization()(x)
    x: KerasTensor = Activation("relu")(x)
    x: KerasTensor = Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding="same", use_bias=False)(x)

    out: KerasTensor = Add()([shortcut, x])

    return out


def custom_resnet(input_shape: tuple[int, int, int], model_name: str) -> Model:
    inputs = Input(shape=input_shape)
    x: KerasTensor = Rescaling(scale=1. / 255)(inputs)

    x: KerasTensor = Conv2D(filters=32, kernel_size=3, strides=2, padding="same", use_bias=False)(x)

    x: KerasTensor = resnet_block(x, filters=64, kernel_size=3, stride=1)
    x: KerasTensor = resnet_block(x, filters=128, kernel_size=3, stride=2)
    x: KerasTensor = resnet_block(x, filters=256, kernel_size=3, stride=2)

    x: KerasTensor = LayerNormalization()(x)
    x: KerasTensor = Activation("relu")(x)

    x: KerasTensor = GlobalAveragePooling2D()(x)
    outputs: KerasTensor = Dense(1, activation="sigmoid")(x)

    return Model(inputs=inputs, outputs=outputs, name=model_name)


def pretrained_resnet152(input_shape: tuple[int, int, int], model_name: str) -> Model:
    base_model = ResNet152(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False

    inputs: KerasTensor = Input(shape=input_shape)
    x: KerasTensor = preprocess_input(inputs)
    x: KerasTensor = base_model(x, training=False)

    x: KerasTensor = GlobalAveragePooling2D()(x)
    x: KerasTensor = Dropout(0.2)(x)
    output: KerasTensor = Dense(1, activation="sigmoid")(x)

    return Model(inputs=inputs, outputs=output, name=model_name)



