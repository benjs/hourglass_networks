import tensorflow as tf

from hourglass.layers import Hourglass, ResidualBottleneck, BulatBlock, LinearLayer, \
    AddFeatureAggregation, ConcatFeatureAggregation, ConcatFeatureAggregation1x1

from tensorflow.keras import Model
from tensorflow.keras.layers import ReLU, Conv2D, BatchNormalization, MaxPool2D, Add, Concatenate


class HourglassModel(Model):
    def __init__(self, n_stacks=8, n_reductions=4, n_heatmaps=16, inner_filters=256,
                 activation=ReLU, bottleneck=ResidualBottleneck, pooling=MaxPool2D, conv_bias=True,
                 aggregation=AddFeatureAggregation, short_preprocessing=False,
                 inner_identity=False):
        super(HourglassModel, self).__init__()

        self.n_stacks = n_stacks
        self.preproc = []

        # CU-Net like preprocessing
        if short_preprocessing:
            self.preproc.extend([
                Conv2D(filters=inner_filters, kernel_size=7, strides=2, padding='same',
                       use_bias=conv_bias, name='conv1_7x7'),
                BatchNormalization(name='bn1'),
                activation(name='act1'),
                pooling(pool_size=(2, 2), strides=2, name='pool_2x2'),
            ])

        # Stacked Hourglass like preprocessing
        else:
            self.preproc.extend([
                Conv2D(filters=inner_filters/4, kernel_size=7, strides=2, padding='same',
                       use_bias=conv_bias, name='conv1_7x7'),
                BatchNormalization(name='bn1'),
                activation(name='act1'),
                bottleneck(filters=inner_filters/2, activation=activation, skip_conv=True,
                           conv_bias=conv_bias, name='resblock1'),
                pooling(pool_size=(2, 2), strides=2, name='pool_2x2'),
                bottleneck(filters=inner_filters/2, activation=activation, conv_bias=conv_bias,
                           name='resblock2'),
                bottleneck(filters=inner_filters, activation=activation, skip_conv=True,
                           conv_bias=conv_bias, name='resblock3')
            ])

        self.hgs = []
        self.bnecks = []
        self.lins = []
        self.preds = []
        self.out_remaps = []
        self.hmap_remaps = []
        self.adds = []

        for i in range(self.n_stacks):
            self.hgs.append(Hourglass(
                n_reductions=n_reductions,
                filters=inner_filters,
                bottleneck=bottleneck,
                aggregation=aggregation,
                pooling=pooling,
                activation=activation,
                conv_bias=conv_bias,
                inner_identity=inner_identity,
                name=f"hourglass{i}"
            ))

            self.bnecks.append(bottleneck(filters=inner_filters, activation=activation,
                               conv_bias=conv_bias, name=f"hg_resblock{i}"))
            self.lins.append(LinearLayer(filters=inner_filters, activation=activation,
                                         conv_bias=conv_bias, name=f"linear{i}"))
            self.preds.append(Conv2D(filters=n_heatmaps, kernel_size=1, use_bias=conv_bias,
                              name=f"conv_pred{i}"))

            # if not last stack
            if i < self.n_stacks-1:
                self.out_remaps.append(Conv2D(filters=inner_filters, kernel_size=1,
                                              use_bias=conv_bias, name=f"out_remap{i}"))
                self.hmap_remaps.append(Conv2D(filters=inner_filters, kernel_size=1,
                                               use_bias=conv_bias, name=f"hmap_remap{i}"))
                self.adds.append(Add(name=f"add{i}"))

        self.concat = Concatenate(axis=0, name='concat')

    # See Figure 4 from Newell et al. https://arxiv.org/pdf/1603.06937.pdf
    def call(self, x):
        out = x
        for layer in self.preproc:
            out = layer(out)

        out_scaled = out
        heatmaps = []

        for i in range(self.n_stacks):
            out = self.hgs[i](out_scaled)
            out = self.bnecks[i](out)
            out = self.lins[i](out)

            heatmap = self.preds[i](out)
            heatmaps.append(heatmap)

            # If not last stack, then
            # aggregate different branches
            if i < self.n_stacks-1:
                out_scaled = self.adds[i](
                    [out_scaled, self.out_remaps[i](out), self.hmap_remaps[i](heatmap)])

        return self.concat(heatmaps)


def StackedHourglass():
    return HourglassModel(n_stacks=8, n_reductions=4, n_heatmaps=16, inner_filters=256,
                          activation=ReLU, bottleneck=ResidualBottleneck, pooling=MaxPool2D,
                          aggregation=AddFeatureAggregation, short_preprocessing=False,
                          conv_bias=True)


def StackedHourglassSmall():
    return HourglassModel(n_stacks=2, n_reductions=4, n_heatmaps=16, inner_filters=144,
                          activation=ReLU, bottleneck=ResidualBottleneck, pooling=MaxPool2D,
                          aggregation=AddFeatureAggregation, short_preprocessing=False,
                          conv_bias=True)


def BulatSmall():
    return HourglassModel(n_stacks=4, n_reductions=4, n_heatmaps=16, inner_filters=144,
                          activation=ReLU, bottleneck=BulatBlock, pooling=MaxPool2D,
                          aggregation=ConcatFeatureAggregation1x1, short_preprocessing=False,
                          conv_bias=False)


def BulatTiny():
    return HourglassModel(n_stacks=2, n_reductions=4, n_heatmaps=16, inner_filters=128,
                          activation=ReLU, bottleneck=BulatBlock, pooling=MaxPool2D,
                          aggregation=ConcatFeatureAggregation1x1, short_preprocessing=False,
                          conv_bias=False)


if __name__ == '__main__':
    from pathlib import Path
    from tensorflow.keras.layers import Input

    vis_dir = Path('visualization')
    vis_dir.mkdir(exist_ok=True)

    # Plot model
    m = StackedHourglassSmall()
    # m = StackedHourglass()
    m.build((None, 256, 256, 3))
    print(m.summary())

    x = Input(shape=(256, 256, 3), batch_size=10, name='Batch of Images')
    m = Model(inputs=[x], outputs=m.call(x))
    tf.keras.utils.plot_model(m, to_file=str(vis_dir/'model.png'), show_shapes=True,
                              show_layer_names=True)

    # Plot layers
    x = Input(shape=(64, 64, 144), batch_size=1, name='Input')
    hg = Hourglass(bottleneck=BulatBlock, aggregation=ConcatFeatureAggregation, conv_bias=False,
                   filters=144)
    m = Model(inputs=[x], outputs=hg.call(x))
    tf.keras.utils.plot_model(m, to_file=str(vis_dir/'hourglass.png'), show_shapes=True,
                              show_layer_names=True)

    x = Input(shape=(64, 64, 256), batch_size=1, name='Input')
    rb = ResidualBottleneck()
    m = Model(inputs=[x], outputs=rb.call(x))
    tf.keras.utils.plot_model(m, to_file=str(vis_dir/'resblock.png'), show_shapes=True,
                              show_layer_names=True)

    x = Input(shape=(64, 64, 128), batch_size=1, name='Input')
    rb = ResidualBottleneck(skip_conv=True)
    m = Model(inputs=[x], outputs=rb.call(x))
    tf.keras.utils.plot_model(m, to_file=str(vis_dir/'resblock_skipconv.png'), show_shapes=True,
                              show_layer_names=True)

    x = Input(shape=(16, 16, 144), batch_size=1, name='Input')
    bb = BulatBlock(filters=144, conv_bias=False)
    m = Model(inputs=[x], outputs=bb.call(x))
    tf.keras.utils.plot_model(m, to_file=str(vis_dir/'bulatblock.png'), show_shapes=True,
                              show_layer_names=True, expand_nested=True)

    x1 = Input(shape=(64, 64, 144), batch_size=1, name='x1')
    x2 = Input(shape=(64, 64, 144), batch_size=1, name='x2')
    cb = ConcatFeatureAggregation(filters=144, conv_bias=False)
    m = Model(inputs=[[x1, x2]], outputs=cb.call([x1, x2]))
    tf.keras.utils.plot_model(m, to_file=str(vis_dir/'concat.png'), show_shapes=True,
                              show_layer_names=True, expand_nested=True)

    ab = AddFeatureAggregation(filters=144, conv_bias=False)
    m = Model(inputs=[[x1, x2]], outputs=ab.call([x1, x2]))
    tf.keras.utils.plot_model(m, to_file=str(vis_dir/'add.png'), show_shapes=True,
                              show_layer_names=True, expand_nested=True)
