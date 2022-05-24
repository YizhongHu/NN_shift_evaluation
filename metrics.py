import tensorflow as tf


def counting_error(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)

    accuracy = tf.reduce_sum(tf.abs(y_true - y_pred), axis=-1)
    accuracy = tf.cast(accuracy, tf.keras.backend.floatx())

    return accuracy


class CountingError(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, name='counting_error', dtype=None):
        super(CountingError, self).__init__(counting_error, name, dtype=dtype)


def false_negative(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)

    diff = y_true - y_pred
    false_negatives = tf.where(diff > 0, x=diff, y=0)

    false_negatives = tf.reduce_sum(false_negatives, axis=-1)
    false_negatives = tf.cast(false_negatives, tf.keras.backend.floatx())

    return false_negatives


class FalseNegative(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, name='false_negative', dtype=None):
        super(FalseNegative, self).__init__(false_negative, name, dtype=dtype)


def false_positive(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)

    diff = y_true - y_pred
    false_positives = tf.where(diff < 0, x=tf.abs(diff), y=0)

    false_positives = tf.reduce_sum(false_positives, axis=-1)
    false_positives = tf.cast(false_positives, tf.keras.backend.floatx())

    return false_positives


class FalsePositive(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, name='false_negative', dtype=None):
        super(FalsePositive, self).__init__(false_positive, name, dtype=dtype)


def duplicate_omission(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)

    indices = tf.reduce_max(y_true, axis=-1) > 1

    y_true = y_true[indices]
    y_pred = y_pred[indices]

    omissions = tf.where((y_true > 1, y_pred >= 1),
                         x=tf.abs(y_true - y_pred), y=0)
    omissions = tf.reduce_sum(omissions, axis=-1)
    omissions = tf.cast(omissions, tf.keras.backend.floatx())

    # diff = y_true - y_pred

    # omissions = diff[diff > 0, y_pred >= 1]
    # omissions = tf.reduce_sum(omissions, axis=-1)
    # omissions = tf.cast(omissions, tf.keras.backend.floatx())

    # detected_duplicates = y_true[y_true > 1, y_pred >= 1]
    # detected_duplicates = tf.reduce_sum(detected_duplicates, axis=-1)
    # detected_duplicates = tf.cast(
    #     detected_duplicates, tf.keras.backend.floatx())

    # return omissions / (detected_duplicates + 1e-6)
    return omissions


class DuplicateOmission(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, name='false_negative', dtype=None):
        super(DuplicateOmission, self).__init__(
            duplicate_omission, name, dtype=dtype)


def duplicate_error(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)

    indices = tf.reduce_max(y_true, axis=-1) > 1

    y_true = y_true[indices]
    y_pred = y_pred[indices]

    accuracy = tf.reduce_sum(tf.abs(y_true - y_pred), axis=-1)
    accuracy = tf.cast(accuracy, tf.keras.backend.floatx())

    return accuracy


class DuplicateError(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, name='duplicate_error', dtype=None):
        super(DuplicateError, self).__init__(
            duplicate_error, name, dtype=dtype)
