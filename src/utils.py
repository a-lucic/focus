import tensorflow as tf
import numpy as np
from tensorflow.losses import Reduction
import os
import errno


def filter_hinge_loss(n_class, mask_vector, feat_input,
					  sigma, temperature, model_fn):
	n_input = feat_input.shape[0]
	if not np.any(mask_vector):
		return np.zeros((n_input, n_class))

	filtered_input = tf.boolean_mask(feat_input, mask_vector)

	if type(sigma) != float or type(sigma) != int:
		sigma = tf.boolean_mask(sigma, mask_vector)
	if type(temperature) != float or type(temperature) != int:
		temperature = tf.boolean_mask(temperature, mask_vector)

	filtered_loss = model_fn(filtered_input, sigma, temperature)

	indices = np.where(mask_vector)[0]
	zero_loss = np.zeros((n_input, n_class))
	hinge_loss = tf.tensor_scatter_nd_add(
		zero_loss,
		indices[:, None],
		filtered_loss,
	)
	return hinge_loss


def safe_euclidean(x, epsilon=10. ** -10, axis=-1):
	return (tf.reduce_sum(x ** 2, axis=axis) + epsilon) ** 0.5


def true_euclidean(x, axis=-1):
	return (tf.reduce_sum(x ** 2, axis=axis)) ** 0.5


def safe_cosine(x1, x2, epsilon=10. ** -10):
	normalize_x1 = tf.nn.l2_normalize(x1, dim=1)
	normalize_x2 = tf.nn.l2_normalize(x2, dim=1)
	dist = tf.losses.cosine_distance(normalize_x1, normalize_x2, axis=-1, reduction=Reduction.NONE) + epsilon
	dist = tf.squeeze(dist)
	dist = tf.cast(dist, tf.float64)
	return dist


def true_cosine(x1: object, x2: object, axis=-1) -> object:
	normalize_x1 = tf.nn.l2_normalize(x1, dim=1)
	normalize_x2 = tf.nn.l2_normalize(x2, dim=1)
	dist = tf.losses.cosine_distance(normalize_x1, normalize_x2, axis=axis, reduction=Reduction.NONE)
	dist = tf.squeeze(dist)
	dist = tf.cast(dist, tf.float64)
	return dist


def safe_l1(x, epsilon=10. ** -10, axis=1):
	return tf.reduce_sum(tf.abs(x), axis=axis) + epsilon


def true_l1(x, axis=1):
	return tf.reduce_sum(tf.abs(x), axis=axis)


def tf_cov(x):
	mean_x = tf.reduce_mean(x, axis=0, keep_dims=True)
	mx = tf.matmul(tf.transpose(mean_x), mean_x)
	vx = tf.matmul(tf.transpose(x), x) / tf.cast(tf.shape(x)[0], tf.float64)
	cov_xx = vx - mx
	return cov_xx

def safe_mahal(x, inv_covar, epsilon=10. ** -10):
	return tf.reduce_sum(tf.multiply(tf.matmul(x + epsilon, inv_covar), x + epsilon), axis=1)


def true_mahal(x, inv_covar):
	return tf.reduce_sum(tf.multiply(tf.matmul(x, inv_covar), x), axis=1)


def mkdir_p(path):
	try:
		os.makedirs(path)
	except OSError as exc:  # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else:
			raise


def safe_open(path, w):
	''' Open "path" for writing, creating any parent directories as needed.'''
	mkdir_p(os.path.dirname(path))
	return open(path, w)
