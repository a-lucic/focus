import dataset
import trees
import tensorflow as tf
import numpy as np
import utils
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
import argparse
import time
import pandas as pd
import json
import re

tf.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('--sigma', type=float, required=False, default=1.)
parser.add_argument('--temperature', type=float, required=False, default=1.)
parser.add_argument('--distance_weight', type=float, required=False, default=0.01)
parser.add_argument('--lr', type=float, required=False, default=0.001)
parser.add_argument('--opt', type=str, required=False, default='adam', help='Options are either adam or gd (as str)')
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--data_name', type=str, required=True)
parser.add_argument('--model_type', type=str, required=True)
parser.add_argument('--distance_function', type=str, required=True)

args = parser.parse_args()
sigma_val = args.sigma
temperature_val = args.temperature
distance_weight_val = args.distance_weight
lr = args.lr
opt = args.opt
model_name = args.model_name
data_name = args.data_name
model_type = args.model_type
distance_function = args.distance_function

assert sigma_val != 0
assert temperature_val >= 0
assert distance_weight_val >= 0
assert lr >= 0
assert opt != ' '
assert model_name != ' '
assert data_name != ' '
assert model_type != ' '

start_time = time.time()
model = joblib.load('../models/{}'.format(model_name))

(feat_columns,
 feat_matrix,
 feat_missing_mask) = dataset.read_tsv_file('../data/{}.tsv'.format(data_name))

feat_matrix = feat_matrix[:, :10]

n_examples = feat_matrix.shape[0]
n_class = len(model.classes_)

# Remove the last column which is the label
feat_input = feat_matrix[:, :-1]
median_values = np.median(feat_input, axis=0)
mad = np.mean(np.abs(feat_input - median_values[None, :]), axis=0)

ground_truth = model.predict(feat_input)
class_index = np.zeros(n_examples, dtype=np.int64)
for i, class_name in enumerate(model.classes_):
	mask = np.equal(ground_truth, class_name)
	class_index[mask] = i
class_index = tf.constant(class_index, dtype=tf.int64)
example_range = tf.constant(np.arange(n_examples, dtype=np.int64))
example_class_index = tf.stack((example_range, class_index), axis=1)

# Include training data to compute covariance matrix for Mahalanobis distance
train_name = re.sub('test', 'train', data_name)
train_data = pd.read_csv('../data/{}.tsv'.format(train_name), sep='\t', index_col=0)
x_train = np.array(train_data.iloc[:, :-1])
covar = utils.tf_cov(x_train)
inv_covar = tf.linalg.inv(covar)

# these will be our perturbed features
perturbed = tf.Variable(initial_value=feat_input,
                        name='perturbed_features',
                        trainable=True)


def convert_model(sigma, temperature):
	return trees.get_prob_classification_forest(
		model, feat_columns, perturbed, sigma=sigma, temperature=temperature)


def prob_from_input(perturbed, sigma, temperature):
	if isinstance(model, DecisionTreeClassifier) == False:
		return trees.get_prob_classification_forest(
			model, feat_columns, perturbed, sigma=sigma, temperature=temperature)
	elif isinstance(model, DecisionTreeClassifier) == True:
		return trees.get_prob_classification_tree(
			model, feat_columns, perturbed, sigma=sigma)


if opt == 'adam':
	optimizer = tf.train.AdamOptimizer(learning_rate=lr)
elif opt == 'gd':
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

output_root = '../hyperparameter_tuning/{}/{}/{}/perturbs_{}_sigma{}_temp{}_dweight{}_lr{}'.format(
	distance_function, data_name, model_type, opt,
	sigma_val, temperature_val, distance_weight_val, lr)

num_iter = 10
sigma = np.full(n_examples, sigma_val)
temperature = np.full(n_examples, temperature_val)
distance_weight = np.full(n_examples, distance_weight_val)
to_optimize = [perturbed]
indicator = np.ones(n_examples)
best_perturb = np.zeros(perturbed.shape)
best_distance = np.full(n_examples, 1000.)  # all distances should be below 1000
perturb_iteration_found = np.full(n_examples, 1000 * num_iter, dtype=np.int64)
average_distance = np.zeros(num_iter)
with utils.safe_open(output_root + '.txt', 'w') as fout:
	fout.write('{} {} {} --sigma={} --temp={} --distance_weight={} --lr={}\n'.format(model_name, opt, distance_function,
	                                                                                 sigma_val, temperature_val,
	                                                                                 distance_weight_val, lr))
	for i in range(num_iter):
		with tf.GradientTape(persistent=True) as t:
			p_model = utils.filter_hinge_loss(
				n_class,
				indicator,
				perturbed,
				sigma,
				temperature,
				prob_from_input
			)
			approx_prob = tf.gather_nd(p_model, example_class_index)

			if distance_function == 'euclidean':
				distance = utils.safe_euclidean(perturbed - feat_input, axis=1)
			elif distance_function == 'cosine':
				distance = utils.safe_cosine(perturbed, feat_input)
			elif distance_function == 'l1':
				distance = utils.safe_l1(perturbed - feat_input)
			elif distance_function == 'mahal':
				distance = utils.safe_mahal(perturbed - feat_input, inv_covar)

			euc_distance = utils.safe_euclidean(perturbed - feat_input, axis=1)
			cos_distance = utils.safe_cosine(perturbed, feat_input)
			l1_distance = utils.safe_l1(perturbed - feat_input)

			hinge_approx_prob = indicator * approx_prob
			loss = tf.reduce_mean(hinge_approx_prob + distance_weight * distance)

			grad = t.gradient(loss, to_optimize)
			optimizer.apply_gradients(zip(grad, to_optimize),
			                          global_step=tf.train.get_or_create_global_step())
			# Make sure perturbed values are between 0 and 1 (inclusive)
			tf.assign(perturbed, tf.math.minimum(1, tf.math.maximum(0, perturbed)))

			if distance_function == 'euclidean':
				true_distance = utils.true_euclidean(perturbed - feat_input, axis=1)
			elif distance_function == 'cosine':
				true_distance = utils.true_cosine(perturbed, feat_input)
			elif distance_function == 'l1':
				true_distance = utils.true_l1(perturbed - feat_input)
			elif distance_function == 'mahal':
				true_distance = utils.true_mahal(perturbed - feat_input, inv_covar)

			cur_predict = model.predict(perturbed.numpy())
			indicator = np.equal(ground_truth, cur_predict).astype(np.float64)
			idx_flipped = np.argwhere(indicator == 0).flatten()

			# get best perturbation so far
			mask_flipped = np.not_equal(ground_truth, cur_predict)  # did prediction flip?

			perturb_iteration_found[idx_flipped] = np.minimum(i + 1, perturb_iteration_found[idx_flipped])

			distance_numpy = true_distance.numpy()
			mask_smaller_dist = np.less(distance_numpy, best_distance)  # is dist < previous best dist?

			temp_dist = best_distance.copy()
			temp_dist[mask_flipped] = distance_numpy[mask_flipped]
			best_distance[mask_smaller_dist] = temp_dist[mask_smaller_dist]

			temp_perturb = best_perturb.copy()
			temp_perturb[mask_flipped] = perturbed[mask_flipped]
			best_perturb[mask_smaller_dist] = temp_perturb[mask_smaller_dist]

			fout.write('iteration: {}\n'.format(i))
			fout.write('loss: {} '.format(loss.numpy()))
			fout.write('unchanged: {} '.format(np.sum(indicator)))
			fout.write('prob: {} '.format(tf.reduce_mean(approx_prob).numpy()))
			fout.write('mean dist: {} '.format(tf.reduce_mean(distance).numpy()))
			fout.write('sigma: {} '.format(np.amax(sigma)))
			fout.write('temp: {}\n'.format(np.amax(temperature)))
			end_time = time.time()

			unchanged_ever = best_distance[best_distance == 1000.]
			counterfactual_examples = best_distance[best_distance != 1000.]
			average_distance[i] = np.mean(counterfactual_examples)

			fout.write('Unchanged ever: {}\n'.format(len(unchanged_ever)))
			if len(unchanged_ever) == 0:

				fout.write(
					'Mean {} dist for cf example v1: {}\n'.format(distance_function, tf.reduce_mean(best_distance)))
				fout.write(
					'Mean {} dist for cf example v2: {}\n'.format(distance_function, np.mean(counterfactual_examples)))

			else:
				fout.write('Not all instances have counterfactual examples!! :(\n')
			fout.write('-------------------------- \n')

	fout.write('Finished in: {}sec \n'.format(np.round(end_time - start_time), 2))

perturb_iteration_found[perturb_iteration_found == 1000 * num_iter] = 0

cf_stats = {'dataset': data_name,
            'model_type': model_type,
            'opt': opt,
            'distance_function': distance_function,
            'sigma': sigma_val,
            'temp': temperature_val,
            'dweight': distance_weight_val,
            'lr': lr,
            'unchanged_ever': len(unchanged_ever),
            'mean_dist': np.mean(counterfactual_examples)

            }


with utils.safe_open(output_root + '_cf_stats.txt', 'w') as gsout:
	json.dump(cf_stats, gsout)

# Output results

df_dist = pd.DataFrame({"id": range(n_examples), "best_distance": best_distance})
df_perturb = pd.DataFrame(best_perturb, columns=feat_columns[:-1])
df = pd.concat([df_dist, df_perturb], axis=1)

df.to_csv(output_root + '.tsv', sep='\t')
print("Finished!! ~{} sec".format(np.round(end_time - start_time), 2))
