
import tensorflow as tf
from const import *
from ops import fc_layer, vgg_block


def input_tensor():
    x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3])
    y = tf.placeholder(tf.float32, [None, 1])
    mask = tf.placeholder(tf.float32, [BATCH_SIZE])

    return x, y, mask


def multi_label_net(x):

    is_training = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)

    x = vgg_block('Block1', x, 3, 32, 2, 1, is_training)
    # print(x.get_shape())

    x = vgg_block('Block2', x, 32, 64, 2, 1, is_training)
    # print(x.get_shape())

    x = vgg_block('Block3', x, 64, 128, 2, 1, is_training)
    # print(x.get_shape())

    x = vgg_block('Block4', x, 128, 256, 3, 1, is_training)
    # print(x.get_shape())

    # color branch
    color_fc1 = fc_layer('color_fc1', x, 256, keep_prob, 'relu')
    color_fc2 = fc_layer('color_fc2', color_fc1, 256, keep_prob, 'relu')
    y_color_conv = fc_layer('color_softmax', color_fc2, 14, keep_prob, 'softmax')

    # shape branch
    shape_fc1 = fc_layer('shape_fc1', x, 256, keep_prob, 'relu')
    shape_fc2 = fc_layer('shape_fc2', shape_fc1, 256, keep_prob, 'relu')
    y_shape_conv = fc_layer('shape_softmax', shape_fc2, 13, keep_prob, 'softmax')

    # opening_type branch
    opening_fc1 = fc_layer('opening_fc1', x, 256, keep_prob, 'relu')
    opening_fc2 = fc_layer('opening_fc2', opening_fc1, 256, keep_prob, 'relu')
    y_opening_conv = fc_layer('opening_softmax', opening_fc2, 6, keep_prob, 'softmax')

    # strap branch
    strap_fc1 = fc_layer('strap_fc1', x, 256, keep_prob, 'relu')
    strap_fc2 = fc_layer('strap_fc2', strap_fc1, 256, keep_prob, 'relu')
    y_strap_conv = fc_layer('strap_softmax', strap_fc2, 5, keep_prob, 'softmax')

    # pattern branch
    pattern_fc1 = fc_layer('pattern_fc1', x, 256, keep_prob, 'relu')
    pattern_fc2 = fc_layer('pattern_fc2', pattern_fc1, 256, keep_prob, 'relu')
    y_pattern_conv = fc_layer('pattern_softmax', pattern_fc2, 8, keep_prob, 'softmax')

    # material branch
    material_fc1 = fc_layer('material_fc1', x, 256, keep_prob, 'relu')
    material_fc2 = fc_layer('material_fc2', material_fc1, 256, keep_prob, 'relu')
    y_material_conv = fc_layer('material_softmax', material_fc2, 7, keep_prob, 'softmax')

    # handle branch
    handle_fc1 = fc_layer('handle_fc1', x, 256, keep_prob, 'relu')
    handle_fc2 = fc_layer('handle_fc2', handle_fc1, 256, keep_prob, 'relu')
    y_handle_conv = fc_layer('handle_softmax', handle_fc2, 5, keep_prob, 'softmax')

    # decoration branch
    decoration_fc1 = fc_layer('decoration_fc1', x, 256, keep_prob, 'relu')
    decoration_fc2 = fc_layer('decoration_fc2', decoration_fc1, 256, keep_prob, 'relu')
    y_decoration_conv = fc_layer('decoration_softmax', decoration_fc2, 8, keep_prob, 'softmax')

    return y_color_conv, y_shape_conv, y_opening_conv, y_strap_conv, y_pattern_conv, y_material_conv, y_handle_conv, y_decoration_conv, is_training, keep_prob


def selective_loss(y_color_conv, y_shape_conv, y_opening_conv, y_strap_conv, y_pattern_conv, y_material_conv, y_handle_conv, y_decoration_conv, y, mask):

    vector_color = tf.constant(0., tf.float32, [BATCH_SIZE])
    vector_shape = tf.constant(1., tf.float32, [BATCH_SIZE])
    vector_opening = tf.constant(2., tf.float32, [BATCH_SIZE])
    vector_strap = tf.constant(3., tf.float32, [BATCH_SIZE])
    vector_pattern = tf.constant(4., tf.float32, [BATCH_SIZE])
    vector_material = tf.constant(5., tf.float32, [BATCH_SIZE])
    vector_handle = tf.constant(5., tf.float32, [BATCH_SIZE])
    vector_decoration = tf.constant(6., tf.float32, [BATCH_SIZE])

    color_mask = tf.cast(tf.equal(mask, vector_color), tf.float32)
    shape_mask = tf.cast(tf.equal(mask, vector_shape), tf.float32)
    opening_mask = tf.cast(tf.equal(mask, vector_opening), tf.float32)
    strap_mask = tf.cast(tf.equal(mask, vector_strap), tf.float32)
    pattern_mask = tf.cast(tf.equal(mask, vector_pattern), tf.float32)
    material_mask = tf.cast(tf.equal(mask, vector_material), tf.float32)
    handle_mask = tf.cast(tf.equal(mask, vector_handle), tf.float32)
    decoration_mask = tf.cast(tf.equal(mask, vector_decoration), tf.float32)

    tf.add_to_collection('color_mask', color_mask)
    tf.add_to_collection('shape_mask', shape_mask)
    tf.add_to_collection('opening_mask', opening_mask)
    tf.add_to_collection('strap_mask', strap_mask)
    tf.add_to_collection('pattern_mask', pattern_mask)
    tf.add_to_collection('material_mask', material_mask)
    tf.add_to_collection('handle_mask', handle_mask)
    tf.add_to_collection('decoration_mask', decoration_mask)

    y_color = tf.slice(y, [0, 0], [BATCH_SIZE, 14])
    y_shape = tf.slice(y, [0, 0], [BATCH_SIZE, 13])
    y_opening = tf.slice(y, [0, 0], [BATCH_SIZE, 6])
    y_strap = tf.slice(y, [0, 0], [BATCH_SIZE, 5])
    y_pattern = tf.slice(y, [0, 0], [BATCH_SIZE, 8])
    y_material = tf.slice(y, [0, 0], [BATCH_SIZE, 7])
    y_handle = tf.slice(y, [0, 0], [BATCH_SIZE, 5])
    y_decoration = tf.slice(y, [0, 0], [BATCH_SIZE, 8])

    tf.add_to_collection('y_color', y_color)
    tf.add_to_collection('y_shape', y_shape)
    tf.add_to_collection('y_opening', y_opening)
    tf.add_to_collection('y_strap', y_strap)
    tf.add_to_collection('y_pattern', y_pattern)
    tf.add_to_collection('y_material', y_material)
    tf.add_to_collection('y_handle', y_handle)
    tf.add_to_collection('y_decoration', y_decoration)

    color_cross_entropy = tf.reduce_sum(
        tf.reduce_sum(-y_color * tf.log(y_color_conv), axis=1) * color_mask) / tf.clip_by_value(
        tf.reduce_sum(color_mask), 1, 1e9)
    shape_cross_entropy = tf.reduce_sum(
        tf.reduce_sum(-y_shape * tf.log(y_shape_conv), axis=1) * shape_mask) / tf.clip_by_value(
        tf.reduce_sum(shape_mask), 1, 1e9)
    opening_cross_entropy = tf.reduce_sum(
        tf.reduce_sum(-y_opening * tf.log(y_opening_conv), axis=1) * opening_mask) / tf.clip_by_value(
        tf.reduce_sum(opening_mask), 1, 1e9)
    strap_cross_entropy = tf.reduce_sum(
        tf.reduce_sum(-y_strap * tf.log(y_strap_conv), axis=1) * strap_mask) / tf.clip_by_value(
        tf.reduce_sum(strap_mask), 1, 1e9)
    pattern_cross_entropy = tf.reduce_sum(
        tf.reduce_sum(-y_pattern * tf.log(y_pattern_conv), axis=1) * pattern_mask) / tf.clip_by_value(
        tf.reduce_sum(pattern_mask), 1, 1e9)
    material_cross_entropy = tf.reduce_sum(
        tf.reduce_sum(-y_material * tf.log(y_material_conv), axis=1) * material_mask) / tf.clip_by_value(
        tf.reduce_sum(material_mask), 1, 1e9)
    handle_cross_entropy = tf.reduce_sum(
        tf.reduce_sum(-y_handle * tf.log(y_handle_conv), axis=1) * handle_mask) / tf.clip_by_value(
        tf.reduce_sum(handle_mask), 1, 1e9)
    decoration_cross_entropy = tf.reduce_sum(
        tf.reduce_sum(-y_decoration * tf.log(y_decoration_conv), axis=1) * decoration_mask) / tf.clip_by_value(
        tf.reduce_sum(decoration_mask), 1, 1e9)

    l2_loss = []
    for var in tf.trainable_variables():
        if var.op.name.find(r'DW') > 0:
            l2_loss.append(tf.nn.l2_loss(var))
    l2_loss = WEIGHT_DECAY * tf.add_n(l2_loss)

    total_loss = color_cross_entropy + shape_cross_entropy + opening_cross_entropy + strap_cross_entropy + pattern_cross_entropy + material_cross_entropy + handle_cross_entropy + decoration_cross_entropy + l2_loss

    return color_cross_entropy, shape_cross_entropy, opening_cross_entropy, strap_cross_entropy, pattern_cross_entropy, material_cross_entropy, handle_cross_entropy, decoration_cross_entropy, l2_loss, total_loss


def train_op(loss, global_step):
    learning_rate = tf.train.exponential_decay(INIT_LR, global_step, DECAY_STEP, DECAY_LR_RATE, staircase=True)
    train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True).minimize(loss,
                                                                                                                   global_step=global_step)
    tf.add_to_collection('learning_rate', learning_rate)
    return train_step
