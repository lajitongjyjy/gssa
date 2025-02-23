import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging

logging.getLogger("tensorflow").setLevel(logging.FATAL)
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from utils import (
    load_dist_mat,
    read_text,
    text_encoder,
    load_dictionary,
    generate_model_save_path,
    calculate_diff,
    calculate_diff_for_array,
)
import numpy as np
from text_cnn import textcnn, compute_acc, compute_loss
from text_rnn import textrnn
from text_birnn import textbirnn
from FGPM_important import FGPM
from Config import config
import pickle
import time
import math

tf.flags.DEFINE_string(
    "data", "ag_news", "Dataset (dbpedia, yahoo_answers, ag_news)"
)

tf.flags.DEFINE_string(
    "nn_type", "textcnn", "Neural network model type (choices: textcnn, textrnn, textbirnn)"
)
tf.flags.DEFINE_string(
    "train_type", "org", "Training method for the model (choices: org, adv)"
)
tf.flags.DEFINE_string("time", None, "Timestamp for the model")
tf.flags.DEFINE_string("step", None, "Checkpoint step for the model")

tf.flags.DEFINE_integer("batch_size", 1000, "Number of samples randomly selected for attack (default: 1000)")
tf.flags.DEFINE_string("recipe", "gssa", "Attack recipe (default: gssa)")
tf.flags.DEFINE_boolean(
    "evaluate_testset", True, "Evaluate the entire test set before attack"
)
tf.flags.DEFINE_boolean(
    "stop_words", True, "Do not modify stop words like prepositions and articles"
)
tf.flags.DEFINE_boolean(
    "save_to_file",
    True,
    "Save adversarial samples and attack results to file <project-dir>/adv_samples/~",
)

tf.flags.DEFINE_float(
    "distance_threshold",
    0.5,
    "Maximum distance for replacement (default: 0.5)",
)
tf.flags.DEFINE_integer(
    "max_candidates",
    4,
    "Number of nearest 'max_candidates' synonyms that satisfy delta constraints during attack (default: 4)",
)

tf.flags.DEFINE_integer("max_iter", 50, "Maximum number of replacements.")
tf.flags.DEFINE_integer("grad_upd_interval", 1, "Gradient update interval")
tf.flags.DEFINE_float(
    "max_perturbed_percent",
    0.15,
    "Upper limit for word replacement rate (default: 0.15)",
)

tf.flags.DEFINE_string("gpu", "0", "GPU to use (default: 0)")

tf.flags.DEFINE_string(
    "data_dir", "./", "Path to hold the input data",
)
tf.flags.DEFINE_string(
    "model_dir", "./", "Path to hold the output data",
)

FLAGS = tf.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
MAX_VOCAB_SIZE = config.num_words[FLAGS.data]


def generate_model_path(model_dir):
    CHECKPOINT_DIR = os.path.join(
        model_dir,
        "runs_%s/%s/checkpoints/"
        % (
            FLAGS.nn_type,
            generate_model_save_path(FLAGS.time, FLAGS.data, FLAGS.train_type),
        ),
    )
    if FLAGS.step == "":
        checkpoint_file = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    else:
        checkpoint_file = CHECKPOINT_DIR + "model_%s" % (FLAGS.step)
    print(checkpoint_file)
    return checkpoint_file


def sample(clean_text_list, labels, sample_num):
    clean_text_list = np.array(clean_text_list)
    labels = np.array(labels)
    np.random.seed(0)
    shuffled_idx = np.arange(0, len(clean_text_list), 1)
    np.random.shuffle(shuffled_idx)
    sampled_idx = shuffled_idx[:sample_num]
    return list(clean_text_list[sampled_idx]), list(labels[sampled_idx])


def encode_convert_to_text(
        perturbed_encoded_text, sample_encoded_text, sample_clean_text, org_inv_dic, dataset
):
    index_overflow = False
    ori_tokens = sample_clean_text.split()
    perturbed_tokens = ori_tokens.copy()
    for i in range(min(len(ori_tokens), config.word_max_len[dataset])):
        if perturbed_encoded_text[i] != sample_encoded_text[i]:
            if perturbed_encoded_text[i] == -1 or perturbed_encoded_text[i] == 0:
                index_overflow = True
                continue
            perturbed_tokens[i] = org_inv_dic[perturbed_encoded_text[i]]
    return index_overflow, " ".join(perturbed_tokens)


def check_index_overflow(
        perturbed_encoded_text, sample_encoded_text, sample_clean_text, dataset
):
    index_overflow = False
    ori_tokens = sample_clean_text.split()
    for i in range(min(len(ori_tokens), config.word_max_len[dataset])):
        if perturbed_encoded_text[i] != sample_encoded_text[i]:
            if perturbed_encoded_text[i] <= 0:
                index_overflow = True
                break
    return index_overflow


def output_flags_log(FLAGS):
    flags_log = ""
    for name, value in FLAGS.__flags.items():
        flags_log += str(name) + ":\t" + str(value.value) + "\n"
    return flags_log


def main(argv=None):
    tf.reset_default_graph()
    session_conf = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(
        config=tf.ConfigProto(gpu_options=session_conf)
    )
    x = tf.placeholder(
        tf.int32, shape=[None, config.word_max_len[FLAGS.data]]
    )
    x_mask = tf.placeholder(
        tf.int32, shape=[None, config.word_max_len[FLAGS.data]]
    )
    y = tf.placeholder(tf.int32, shape=[None])
    x_org = tf.placeholder(
        tf.int32, shape=[None, config.word_max_len[FLAGS.data]]
    )

    if FLAGS.nn_type == "textcnn":
        _, embedded_chars, predictions, scores = textcnn(x, 1.0, FLAGS.data)
    elif FLAGS.nn_type == "textrnn":
        _, embedded_chars, predictions, scores = textrnn(x, 1.0, FLAGS.data)
    elif FLAGS.nn_type == "textbirnn":
        _, embedded_chars, predictions, scores = textbirnn(x, 1.0, FLAGS.data)
    saver = tf.train.Saver()
    checkpoint_file = generate_model_path(FLAGS.model_dir)
    saver.restore(sess, checkpoint_file)
    org_dic, org_inv_dic = load_dictionary(FLAGS.data, MAX_VOCAB_SIZE, FLAGS.data_dir)
    print("The dictionary has %d words." % len(org_dic))
    if FLAGS.train_type == "org" or FLAGS.train_type == "adv":
        dist_mat = load_dist_mat(FLAGS.data, MAX_VOCAB_SIZE, FLAGS.data_dir)
    else:
        raise NotImplementedError
    if FLAGS.stop_words:
        print("Enable stop words.")
        for stop_word in config.stop_words:
            if stop_word in org_dic:
                dist_mat[org_dic[stop_word], :, :] = 0
    dist_mat = dist_mat[:, : FLAGS.max_candidates, :]
    clean_texts, labels = read_text("%s/test" % FLAGS.data, data_dir=FLAGS.data_dir)
    encoded_texts, _ = text_encoder(
        clean_texts, org_dic, config.word_max_len[FLAGS.data]
    )
    if FLAGS.evaluate_testset:
        print("Model accuracy on test set:")
        correct_predict_count = 0
        sample_num = len(clean_texts)
        for i in range(math.ceil(sample_num / 500)):
            pred = sess.run(
                predictions,
                feed_dict={
                    x: encoded_texts[i * 500: (i + 1) * 500],
                    y: labels[i * 500: (i + 1) * 500],
                },
            )
            for j in range(len(pred)):
                if pred[j] == labels[i * 500 + j]:
                    correct_predict_count += 1
        testset_acc = correct_predict_count / sample_num
        print(
            correct_predict_count, "/", sample_num, "=", testset_acc,
        )
    print("Sample ", FLAGS.batch_size, "samples to attack...")
    sample_clean_texts, sample_labels = sample(clean_texts, labels, FLAGS.batch_size)
    sample_encoded_texts, sample_encoded_texts_mask = text_encoder(
        sample_clean_texts, org_dic, config.word_max_len[FLAGS.data]
    )

    start_attack_time = time.time()

    if FLAGS.recipe == "gssa":
        perturbed_x, wrong_predict_state_tensor, adv_labels = FGPM(
            x,
            y,
            x_mask,
            FLAGS.data,
            FLAGS.nn_type,
            FLAGS.max_iter,
            config.num_classes[FLAGS.data],
            dist_mat,
            FLAGS.grad_upd_interval,
            dis_threshold=FLAGS.distance_threshold,
            sn=FLAGS.max_candidates,
            max_perturbed_percent=FLAGS.max_perturbed_percent,
            xs_org=x_org,
        )

        print("gssa Attack: Computation graph created!")

        perturbed_encoded_texts = []
        wrong_predict_state = []

        res_x, res_state, res_adv_labels = sess.run(
            [perturbed_x, wrong_predict_state_tensor, adv_labels],
            feed_dict={
                x: sample_encoded_texts[: FLAGS.batch_size],
                y: sample_labels[: FLAGS.batch_size],
                x_mask: sample_encoded_texts_mask[: FLAGS.batch_size],
                x_org: sample_encoded_texts[: FLAGS.batch_size],
            },
        )

        perturbed_encoded_texts.extend(res_x)
        wrong_predict_state.extend(res_state)

        print("gssa Attack time: ", time.time() - start_attack_time)

        for i in range(FLAGS.batch_size):
            print("Progress: ", i, "/", FLAGS.batch_size)
            sample_clean_text = sample_clean_texts[i]
            sample_label = sample_labels[i]
            sample_encoded_text = sample_encoded_texts[i]
            index_overflow, perturbed_text = encode_convert_to_text(
                perturbed_encoded_texts[i],
                sample_encoded_text,
                sample_clean_text,
                org_inv_dic,
                FLAGS.data,
            )
            if index_overflow:
                continue
            if FLAGS.save_to_file:
                adv_sample_file = open(
                    "%s/adv_samples/%s.txt"
                    % (FLAGS.data_dir, str(sample_label)),
                    "a",
                )
                adv_sample_file.write(perturbed_text + "\n")
                adv_sample_file.close()
        print("FGPM Attack Finished!")
    else:
        print("Attack recipe not implemented!")
