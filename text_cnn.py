# 使用 TensorFlow 的兼容版本1
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()  # 禁用 TensorFlow v2 的行为
import numpy as np
from Config import config  # 从 Config 模块导入配置


# 定义 TextCNN 函数
def textcnn(input_x, dropout_keep_prob, dataset, reuse=False):
    """
    文本分类的 CNN。
    使用嵌入层，然后是三个卷积 + 最大池化层，一个 dropout 层和一个全连接层。
    """

    # 从配置中获取数据集的参数
    sequence_length = config.word_max_len[dataset]  # 句子的最大长度
    num_classes = config.num_classes[dataset]  # 类别数量
    vocab_size = config.num_words[dataset]  # 词汇表的大小

    # 定义参数
    embedding_size = 300  # 嵌入向量的大小
    filter_sizes = [3, 4, 5]  # 卷积滤波器的大小
    num_filters = 128  # 每种滤波器的数量

    # 在 "test" 变量范围下定义网络结构
    with tf.variable_scope("test", reuse=reuse):
        # 嵌入层
        with tf.variable_scope("embedding", reuse=reuse):
            # 创建嵌入矩阵
            embeddings = tf.get_variable(
                initializer=tf.random_uniform([vocab_size + 1, embedding_size], -1.0, 1.0),
                name="W",
                trainable=True,
            )
            # 查找输入数据的嵌入向量
            embedded_chars = tf.nn.embedding_lookup(
                embeddings, input_x, name="embedded_chars"
            )  # [None, sequence_length, embedding_size]
            # 扩展嵌入向量的维度，为卷积操作做准备
            embedded_chars_expanded = tf.expand_dims(
                embedded_chars, -1
            )  # [None, sequence_length, embedding_size, 1]

        # 为每种滤波器大小创建卷积 + 最大池化层
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size, reuse=reuse):
                # 卷积层
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.get_variable(
                    initializer=tf.truncated_normal(filter_shape, stddev=0.1),
                    name="W",
                )
                b = tf.get_variable(
                    initializer=tf.constant(0.1, shape=[num_filters]), name="b"
                )
                conv = tf.nn.conv2d(
                    embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv",
                )
                # 非线性激活函数
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # 最大池化操作
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool",
                )
                pooled_outputs.append(pooled)

        # 合并所有的池化输出
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # 添加 dropout 层
        with tf.variable_scope("dropout", reuse=reuse):
            h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob, name="text_vector")

        # 输出层
        with tf.variable_scope("output", reuse=reuse):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.keras.initializers.glorot_normal()
            )
            b = tf.get_variable(initializer=tf.constant(0.1, shape=[num_classes]), name="b")
            scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")  # 得分
            predictions = tf.argmax(scores, 1, name="predictions", output_type=tf.int32)  # 预测

    return embeddings, embedded_chars, predictions, scores

#计算给定的逻辑和真实标签之间的损失
def compute_loss(logits, input_y, num_classes):
    losses = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(input_y, depth=num_classes), logits=logits
        )
    )
    return losses

#根据预测和实际标签计算准确率
def compute_acc(predictions, input_y):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(input_y, predictions), tf.float32))
    return accuracy
