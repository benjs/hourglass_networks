import tensorflow as tf

from tensorflow.keras.layers import Layer, Conv2D, ReLU, BatchNormalization, MaxPool2D, \
    UpSampling2D, Add, Concatenate


# Residual bottleneck layer as used by Newell et al. https://arxiv.org/pdf/1603.06937.pdf
# originally from He et al. https://arxiv.org/pdf/1512.03385.pdf
class ResidualBottleneck(Layer):
    def __init__(self, filters=256, skip_conv=False, bneck_ratio=0.5, conv_bias=True,
                 activation=ReLU, **kwargs):
        super(ResidualBottleneck, self).__init__(**kwargs)

        self.need_skip_conv = skip_conv

        self.bn1 = BatchNormalization(name='bn1')
        self.act1 = activation(name='act1')
        self.conv1 = Conv2D(filters=filters*bneck_ratio, kernel_size=1, use_bias=conv_bias,
                            name='conv1_1x1')

        self.bn2 = BatchNormalization(name='bn2')
        self.act2 = activation(name='act2')
        self.conv2 = Conv2D(filters=filters*bneck_ratio, kernel_size=3, padding='same',
                            use_bias=conv_bias, name='conv2_3x3')

        self.bn3 = BatchNormalization(name='bn3')
        self.act3 = activation(name='act3')
        self.conv3 = Conv2D(filters=filters, kernel_size=1, use_bias=conv_bias, name='conv3_1x1')

        # Skip conv required when number of input filters does
        # not match number of output filters
        if self.need_skip_conv:
            self.skip_conv = Conv2D(filters=filters, kernel_size=1, use_bias=conv_bias,
                                    name='skip_conv_1x1')

        self.add = Add(name='add')

    def call(self, x):
        out = self.bn1(x)
        out = self.act1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv3(out)

        if self.need_skip_conv:
            skip = self.skip_conv(x)
        else:
            skip = x

        out = self.add([out, skip])
        return out


# Custom layer to support batched tf.math.multiply
class SoftGatedSkipConnection(Layer):
    def __init__(self, filters=256, skip_conv=False, conv_bias=False, **kwargs):
        super(SoftGatedSkipConnection, self).__init__(**kwargs)

        self.filters = filters
        self.need_skip_conv = skip_conv

        # The authors didnt mention skip connections, this is a test
        if self.need_skip_conv:
            self.skip_conv = Conv2D(filters=filters, kernel_size=1, use_bias=conv_bias,
                                    name='skip_conv_1x1')

    def build(self, input_shape):
        self.perchannel = self.add_weight(shape=(1, 1, 1, int(self.filters)), initializer='ones',
                                          trainable=True, name='perchannel_weights')

    def call(self, x):
        if self.need_skip_conv:
            x = self.skip_conv(x)

        return tf.math.multiply(x, self.perchannel)


class Identity(Layer):
    def __init__(self, **kwargs):
        super(Identity, self).__init__(**kwargs)

    def call(self, x):
        return x


class BulatBlock(Layer):
    def __init__(self, filters=256, activation=ReLU, conv_bias=False, skip_conv=None, **kwargs):
        super(BulatBlock, self).__init__(**kwargs)

        self.sgsc = SoftGatedSkipConnection(filters=filters, skip_conv=skip_conv,
                                            conv_bias=conv_bias, name='sgsc')

        self.bn1 = BatchNormalization(name='bn1')
        self.act1 = activation(name='act1')
        self.conv1 = Conv2D(filters=filters/2, kernel_size=3, padding='same', use_bias=conv_bias,
                            name='conv1_3x3')

        self.bn2 = BatchNormalization(name='bn2')
        self.act2 = activation(name='act2')
        self.conv2 = Conv2D(filters=filters/4, kernel_size=3, padding='same', use_bias=conv_bias,
                            name='conv2_3x3')

        self.bn3 = BatchNormalization(name='bn3')
        self.act3 = activation(name='act3')
        self.conv3 = Conv2D(filters=filters/4, kernel_size=3, padding='same', use_bias=conv_bias,
                            name='conv3_3x3')

        self.concat = Concatenate(axis=-1, name='concat')
        self.add = Add(name='add')

    def call(self, x):
        out = self.bn1(x)
        out = self.act1(out)
        out1 = self.conv1(out)

        out = self.bn2(out1)
        out = self.act2(out)
        out2 = self.conv2(out)

        out = self.bn3(out2)
        out = self.act3(out)
        out3 = self.conv3(out)

        out_concat = self.concat([out1, out2, out3])
        out_skip = self.sgsc(x)

        out = self.add([out_concat, out_skip])
        return out


class AddFeatureAggregation(Layer):
    def __init__(self, filters=None, conv_bias=None, **kwargs):
        super(AddFeatureAggregation, self).__init__(**kwargs)
        self.add = Add(name='add')

    def call(self, x):
        x_down = x[0]
        x_up = x[1]

        return self.add([x_down, x_up])


# See Bulat et al.
class ConcatFeatureAggregation(Layer):
    def __init__(self, filters=256, conv_bias=True, **kwargs):
        super(ConcatFeatureAggregation, self).__init__(**kwargs)

        self.concat = Concatenate(axis=-1, name='concat')
        self.conv = Conv2D(filters=filters, kernel_size=3, padding='same', use_bias=conv_bias,
                           name='conv_3x3')

    def call(self, x):
        x_down = x[0]
        x_up = x[1]

        out = self.concat([x_down, x_up])  # 2*N channels
        out = self.conv(out)  # 2*N channels -> N channels
        return out


class ConcatFeatureAggregation1x1(ConcatFeatureAggregation):
    def __init__(self, filters=256, conv_bias=True, **kwargs):
        super(ConcatFeatureAggregation1x1, self).__init__(**kwargs)

        self.conv = Conv2D(filters=filters, kernel_size=1, use_bias=conv_bias, name='conv_1x1')


class LinearLayer(Layer):
    def __init__(self, filters=256, activation=ReLU, conv_bias=True, **kwargs):
        super(LinearLayer, self).__init__(**kwargs)
        self.conv = Conv2D(filters=filters, kernel_size=1, use_bias=conv_bias, name='conv_1x1')
        self.bn = BatchNormalization(name='bn')
        self.activation = activation()

    def call(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        return out


class Hourglass(Layer):
    def __init__(self, n_reductions=4, filters=256, bottleneck=ResidualBottleneck, conv_bias=True,
                 aggregation=AddFeatureAggregation, pooling=MaxPool2D, activation=ReLU,
                 inner_identity=False, **kwargs):
        super(Hourglass, self).__init__(**kwargs)

        self.activation = activation()

        # Downsample path
        self.pool = pooling(pool_size=(2, 2), strides=2, name='pool_2x2')
        self.bneck_down = bottleneck(filters=filters, activation=activation, conv_bias=conv_bias,
                                     name='resblock_down')

        if n_reductions > 1:
            # Rekursively go deeper
            self.inner = Hourglass(
                n_reductions=n_reductions-1,
                filters=filters,
                bottleneck=bottleneck,
                aggregation=aggregation,
                pooling=pooling,
                activation=activation,
                conv_bias=conv_bias,
            )
        else:
            if inner_identity:
                # Bulat et al.:
                self.inner = Identity()
            else:
                # Original Newell et al.:
                self.inner = bottleneck(filters=filters, activation=activation,
                                        conv_bias=conv_bias, name='resblock_inner')

        # Upsample path
        self.bneck_up = bottleneck(filters=filters, activation=activation, conv_bias=conv_bias,
                                   name='resblock_up')
        self.upsample = UpSampling2D(size=(2, 2), interpolation='nearest', name='upsample_2x2')

        # Aggregation branch
        self.bneck_fagg = bottleneck(filters=filters, activation=activation, conv_bias=conv_bias,
                                     name='resblock_agg')

        # Connection of aggregation branch and deeper branch
        self.fagg = aggregation(filters=filters, conv_bias=conv_bias, name='aggregation')

    def call(self, x):
        # Going deeper
        out = self.pool(x)
        out = self.bneck_down(out)
        out = self.inner(out)
        out = self.bneck_up(out)
        out = self.upsample(out)

        # Aggregation path on same "depth" in network
        out_agg = self.bneck_fagg(x)

        # Connect both branches
        out = self.fagg([out_agg, out])
        return out
