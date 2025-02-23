# -*- coding: UTF-8 -*-

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from Config import config

def textbirnn(input_x, dropout_keep_prob, dataset, reuse=False):
    """
    A Bi-directional RNN for text classification.
    Uses an embedding layer, followed by a bi-directional LSTM layer, a dropout layer and a fully-connected layer.
    用于文本分类的双向循环神经网络（RNN）模型。
    包括嵌入层、双向LSTM层、Dropout层和输出层。
    """
    # 从配置文件中获取类别数和词汇表大小
    num_classes = config.num_classes[dataset]
    vocab_size = config.num_words[dataset]
    embedding_size = 300

    # 嵌入层
    with tf.variable_scope("embedding", reuse=reuse):
        # 创建词嵌入矩阵，可训练
        embeddings = tf.get_variable("W",
            initializer=tf.random_uniform([vocab_size+1, embedding_size], -1.0, 1.0),
            trainable=True)
        # 将输入序列映射到嵌入空间
        embedded_chars = tf.nn.embedding_lookup(embeddings, input_x, name="embedded_chars")  # [None, sequence_length, embedding_size]

    # 创建每个RNN层的双向LSTM层
    with tf.variable_scope('bilstm', reuse=reuse):
        cell_fun = tf.nn.rnn_cell.BasicLSTMCell

        def get_bi_cell():
            fw_cell = cell_fun(128, state_is_tuple=True) #前向方向的LSTM单元
            bw_cell = cell_fun(128, state_is_tuple=True) #后向方向的LSTM单元
            return fw_cell, bw_cell

        # Bi-lstm layer双向LSTM层
        fw_cell, bw_cell = get_bi_cell()
        outputs, last_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, embedded_chars, dtype=tf.float32)
        outputs = tf.concat(outputs, axis=2)
        output = tf.reduce_mean(outputs, axis=1)

    # 添加Dropout，防止过拟合
    with tf.variable_scope("dropout", reuse=reuse):
        rnn_drop = tf.nn.dropout(output, dropout_keep_prob)

    # 最终（未归一化）得分和预测
    with tf.variable_scope("output", reuse=reuse):
        # 创建权重和偏差变量
        W = tf.get_variable(
            "W",
            shape=[128*2, num_classes],
            #initializer=tf.contrib.layers.xavier_initializer()
            initializer = tf.compat.v1.keras.initializers.glorot_normal(),
        )
        b = tf.get_variable("b", initializer=tf.constant(0.1, shape=[num_classes]))
        # 计算得分（未归一化）
        scores = tf.nn.xw_plus_b(rnn_drop, W, b, name="scores")
        # 预测最可能的类别
        predictions = tf.argmax(scores, 1, name="predictions", output_type=tf.int32)

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