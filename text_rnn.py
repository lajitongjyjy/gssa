# 使用TensorFlow的兼容版本1
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from Config import config  # 从Config模块导入配置


# 定义TextRNN函数
def textrnn(input_x, dropout_keep_prob, dataset, reuse=False):
    """
    一个用于文本分类的RNN。
    使用嵌入层，接着是三个LSTM层、一个dropout层和一个全连接层。
    """

    num_classes = config.num_classes[dataset]  # 获取类别数量
    vocab_size = config.num_words[dataset]  # 获取词汇表的大小
    embedding_size = 300  # 嵌入向量的大小

    # 嵌入层
    with tf.variable_scope("embedding", reuse=reuse):
        embeddings = tf.get_variable(
            "W",
            initializer=tf.random_uniform([vocab_size + 1, embedding_size], -1.0, 1.0),
            trainable=True,  # 设置为可训练
        )
        # 查找输入数据的嵌入向量
        embedded_chars = tf.nn.embedding_lookup(
            embeddings, input_x, name="embedded_chars"
        )  # [None, sequence_length, embedding_size]

    # 为每个RNN层创建一个LSTM单元
    with tf.variable_scope("lstm", reuse=reuse):
        ####################################弃用
        cell_fun = tf.compat.v1.nn.rnn_cell.LSTMCell
        def get_a_cell():
            cell_tmp = cell_fun(128, state_is_tuple=True)  # 创建一个128个单元的LSTM
            return cell_tmp

        # 堆叠多个LSTM层
        cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(3)])  # 创建三层的LSTM
        outputs, last_state = tf.nn.dynamic_rnn(
            cell, embedded_chars, dtype=tf.float32
        )  # 使用dynamic_rnn进行计算
        output = tf.reduce_mean(outputs, axis=1)  # 计算outputs的均值作为输出

    # 添加dropout层
    with tf.variable_scope("dropout", reuse=reuse):
        rnn_drop = tf.nn.dropout(output, dropout_keep_prob)  # 对output进行dropout操作

    # 输出层
    with tf.variable_scope("output", reuse=reuse):
        W = tf.get_variable(
            "W",
            shape=[128, num_classes],  # 定义权重矩阵的形状
            initializer = tf.compat.v1.keras.initializers.glorot_normal(),
        )
        b = tf.get_variable("b", initializer=tf.constant(0.1, shape=[num_classes]))  # 定义偏置
        scores = tf.nn.xw_plus_b(rnn_drop, W, b, name="scores")  # 计算得分
        predictions = tf.argmax(scores, 1, name="predictions", output_type=tf.int32)  # 根据得分进行预测

    return embeddings, embedded_chars, predictions, scores  # 返回嵌入矩阵、嵌入字符、预测和得分



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