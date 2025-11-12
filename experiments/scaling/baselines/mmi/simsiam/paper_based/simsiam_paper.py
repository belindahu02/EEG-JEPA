import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, MaxPooling1D, Dropout
from backbones import resnetblock_final

# Projector
def proTian(inputs):
    """Projector: Dense(2048, BN, ReLU) → Dense(512, BN)"""
    x = layers.Dense(2048, use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    outputs = layers.Dense(512, use_bias=False)(x)
    outputs = BatchNormalization()(outputs)
    return outputs


# Predictor
def predTian(inputs):
    """Predictor: 8196×3 → 4096 → 2048 → 512"""
    x = layers.Dense(8196, use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = layers.Dense(8196, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = layers.Dense(8196, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = layers.Dense(4096, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = layers.Dense(2048, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    outputs = layers.Dense(512, use_bias=False)(x)
    outputs = BatchNormalization()(outputs)
    return outputs


# Encoder
def get_encoder(frame_size, ftr, mlp_s, origin):
    ks = 3
    k = (48, 96)  # MMI dataset config

    inputs = layers.Input((frame_size, ftr), name='input')

    # First conv: 48 filters
    x = Conv1D(filters=k[0], kernel_size=ks, strides=1, padding='same',
               kernel_regularizer=tf.keras.regularizers.L2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=4, strides=4)(x)
    x = Dropout(rate=0.1)(x)

    # ResNet block: 96 filters
    backbone_features = resnetblock_final(x, CR=k[1], KS=ks)

    # Global pooling to flatten (mentioned in Figure 3)
#    backbone_features = layers.GlobalMaxPooling1D()(backbone_features)

    # Projection head
    projected = proTian(backbone_features)

    encoder = tf.keras.Model(inputs, projected, name="encoder")
    backbone = tf.keras.Model(inputs, backbone_features, name="backbone")
    return encoder, backbone


# Predictor wrapper
def get_predictor(mlp_s, origin):
    inputs = layers.Input((512,))  # matches projector output
    outputs = predTian(inputs)
    return tf.keras.Model(inputs, outputs, name="predictor")


# Loss function
def compute_loss(p, z):
    """Negative cosine similarity"""
    z = tf.stop_gradient(z)
    # Normalize vectors
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    # Negative cosine similarity
    return -tf.reduce_mean(tf.reduce_sum(p * z, axis=1))


class Contrastive(tf.keras.Model):
    def __init__(self, encoder, predictor, backbone=None):
        super(Contrastive, self).__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.backbone = backbone
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        ds_one, ds_two = data

        with tf.GradientTape() as tape:
            z1, z2 = self.encoder(ds_one), self.encoder(ds_two)
            p1, p2 = self.predictor(z1), self.predictor(z2)
            # Symmetric loss with stop-gradient
            loss = compute_loss(p1, z2) / 2 + compute_loss(p2, z1) / 2

        learnable_params = (
            self.encoder.trainable_variables + self.predictor.trainable_variables
        )
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
