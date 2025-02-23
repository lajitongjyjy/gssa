import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from text_cnn import textcnn, compute_loss
from text_rnn import textrnn, compute_loss
from text_birnn import textbirnn
from Config import Config

def find_synonym(xs, dist_mat, batch_size, word_max_len, threshold=0.5):
    xs = tf.expand_dims(xs, -1)
    synonyms = tf.gather_nd(dist_mat[:, :, 0], xs)
    synonyms_dist = tf.gather_nd(dist_mat[:, :, 1], xs)
    synonyms = tf.where(synonyms_dist <= threshold, synonyms, tf.zeros_like(synonyms))
    synonyms = tf.where(synonyms >= 0, synonyms, tf.zeros_like(synonyms))
    return synonyms
def gssa(
        xs,
        ys,
        xs_mask,
        dataset,
        model,
        max_iter,  
        num_classes,
        dist_mat,
        dis_threshold=0.5, 
        max_perturbed_percent=0.15,  
        embedding_size=300,
        xs_org=None,
):

    adv_xs = tf.identity(xs)
    if xs_org is None:
        xs_org = xs
    batch_size, word_max_len = tf.unstack(tf.shape(xs))

    modified_mask = tf.zeros_like(xs_mask, dtype=tf.int32)
    words_num = tf.reduce_sum(xs_mask, axis=-1)
    synonyms = tf.cast(
        find_synonym(xs_org, dist_mat, batch_size, word_max_len, dis_threshold),
        tf.int32,
    )

    query = eval(model)

    embeddings, embedded_chars, predictions, logits = query(
        adv_xs, 1.0, dataset, reuse=True
    )
    loss = compute_loss(logits, ys, num_classes)
    Jacobian = tf.gradients(loss, embedded_chars)[0] 
    synonyms_embed = tf.gather_nd(embeddings, tf.expand_dims(synonyms, -1))
    xs_embed = tf.expand_dims(embedded_chars, -2)
    Jacobian = tf.expand_dims(Jacobian, -2)

    scores = tf.reduce_sum(
        tf.multiply(synonyms_embed - xs_embed, Jacobian), axis=-1
    )
    sorted_scores = tf.argsort(
        scores,
        axis=-1,
        direction='DESCENDING',
        stable=True
    )

    for pos in tf.range(word_max_len):
        current_idx = sorted_scores[:, pos]
        candidate_mask = tf.one_hot(current_idx, depth=word_max_len, dtype=tf.bool)
        current_synonyms = tf.boolean_mask(synonyms, candidate_mask)
        new_adv_xs = tf.where(
            candidate_mask,
            current_synonyms[:, 0], 
            adv_xs
        )
        _, _, new_pred, _ = query(new_adv_xs, 1.0, dataset, reuse=True)
        modified_count = modified_mask + tf.cast(candidate_mask, tf.int32)
        perturb_ratio = tf.cast(modified_count, tf.float32) / tf.cast(words_num, tf.float32)
        update_mask = tf.logical_and(
            tf.equal(predictions, ys), 
            tf.logical_and(
                tf.equal(new_pred, ys), 
                tf.less(perturb_ratio, max_perturbed_percent)
            )
        )
        adv_xs = tf.where(
            tf.expand_dims(update_mask, -1),
            new_adv_xs,
            adv_xs
        )
        modified_mask = tf.where(
            update_mask,
            modified_count,
            modified_mask
        )
        success_mask = tf.not_equal(new_pred, ys)
        if tf.reduce_any(success_mask):
            return adv_xs, success_mask, predictions 
    return adv_xs, tf.not_equal(predictions, ys), predictions 
