import sys, os, csv
import numpy as np
import graphviz
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
plt.rcParams.update({'font.size': 128})

plot_stuff = False
covariance_matrix = False
covariance_matrix_full = False
DecisionTree = True

top_recos = 3
test_size = .2

first_data_line = 4

def comput_accuracy_matrix(binary_pred_matrix, label_matrix):
	combined_mat = binary_pred_matrix + label_matrix
	combined_mat[combined_mat < 2] = 0
	combined_mat /= 2

	col_sums = np.sum(combined_mat, axis=0)

	return combined_mat, np.mean(col_sums)

def plot_it2(preds, y_test, label):
	binary_preds = preds.copy()

	for i in range(binary_preds.shape[1]):
		col_thresh = np.sort(binary_preds[:,i])[-top_recos]
		col_dummy = binary_preds[:,i]
		col_dummy[col_dummy < col_thresh] = 0
		col_dummy[col_dummy > 0] = 1
		binary_preds[:,i] = col_dummy

	plt.figure(figsize=(240,240))

	plt.subplot(221)
	plt.imshow(preds, vmin=0, vmax=1, cmap='magma')
	plt.xticks(np.arange(preds.shape[1]), np.arange(preds.shape[1]))
	plt.yticks(np.arange(preds.shape[0]), rec_cols)
	plt.colorbar(shrink=.25, pad=.01)
	plt.title('pred matrix')
	plt.xlabel('sample')
	plt.grid()

	plt.subplot(222)
	plt.imshow(binary_preds, vmin=0, vmax=1, cmap='magma')
	plt.xticks(np.arange(preds.shape[1]), np.arange(preds.shape[1]))
	plt.yticks(np.arange(preds.shape[0]), rec_cols)
	plt.colorbar(shrink=.25, pad=.01)
	plt.xlabel('sample')
	plt.title('binary pred matrix')
	plt.grid()

	plt.subplot(223)
	plt.imshow(np.transpose(y_test), vmin=0, vmax=1, cmap='magma')
	plt.xticks(np.arange(preds.shape[1]), np.arange(preds.shape[1]))
	plt.yticks(np.arange(preds.shape[0]), rec_cols)
	plt.colorbar(shrink=.25, pad=.01)
	plt.xlabel('sample')
	plt.title('label recommendations')
	plt.grid()

	corresp_mat, corresp_acc = comput_accuracy_matrix(binary_preds, np.transpose(y_test))
	plt.subplot(224)
	plt.imshow(corresp_mat, vmin=0, vmax=1, cmap='magma')
	plt.xticks(np.arange(preds.shape[1]), np.arange(preds.shape[1]))
	plt.yticks(np.arange(preds.shape[0]), rec_cols)
	plt.colorbar(shrink=.25, pad=.01)
	plt.xlabel('sample')
	plt.title('correspondances  |  avg. TPs/sample:' + str(np.round(corresp_acc, 2)))
	plt.grid()
	plt.savefig('preds_' + str(label) + '.pdf')
	plt.close()

def plot_it(preds, y_test, label):
	binary_preds = preds.copy()

	for i in range(binary_preds.shape[1]):
		col_thresh = np.sort(binary_preds[:,i])[-top_recos]
		col_dummy = binary_preds[:,i]
		col_dummy[col_dummy < col_thresh] = 0
		col_dummy[col_dummy > 0] = 1
		binary_preds[:,i] = col_dummy


	corresp_mat, avg_TP = comput_accuracy_matrix(binary_preds, np.transpose(y_test))

	false_negatives = np.zeros(binary_preds.shape)
	false_positives = np.zeros(binary_preds.shape)
	true_negatives = np.zeros(binary_preds.shape)

	for row in range(binary_preds.shape[0]):
		for col in range(binary_preds.shape[1]):

			if corresp_mat[row,col] != 1:

				if binary_preds[row,col] == 1 and y_test[col,row] == 0: false_positives[row,col] = 1
				elif binary_preds[row,col] == 0 and y_test[col,row] == 1: false_negatives[row,col] = 1
				elif binary_preds[row,col] == 0 and y_test[col,row] == 0: true_negatives[row,col] = 1


	num_positives = np.mean(np.sum(np.transpose(y_test), axis=0))
	num_negatives = np.mean(np.sum(np.transpose(np.abs(1-y_test)), axis=0))

	avg_TP = np.mean(np.sum(corresp_mat, axis=0))
	avg_FP = np.mean(np.sum(false_positives, axis=0))
	avg_FN = np.mean(np.sum(false_negatives, axis=0))
	avg_TN = np.mean(np.sum(true_negatives, axis=0))

	recall = avg_TP / (avg_TP + avg_FN) 
	FPR = avg_FP / (avg_TP + avg_FN) 
	TPR = avg_TP / (avg_TP + avg_FN) 
	FNR = avg_FN / (avg_TN + avg_FP) 
	TNR = avg_TN / (avg_TN + avg_FP) 
	precision = avg_TP / (avg_TP + avg_FP)
	accuracy = (avg_TP + avg_TN) / (avg_TP + avg_TN + avg_FP + avg_FN)



	plt.figure(figsize=(240,240))

	plt.rcParams.update({'font.size': 256})
	plt.suptitle('recall: ' + str(np.round(100*recall)) + '% | precision: ' + str(np.round(100*precision)) + '% | accuracy: ' + str(np.round(100*accuracy)) + '%')
	plt.rcParams.update({'font.size': 128})

	plt.subplot(221)
	plt.imshow(corresp_mat, vmin=0, vmax=1, cmap='magma', aspect=np.round(preds.shape[1]/preds.shape[0]))
	plt.rcParams.update({'font.size': 32})
	plt.xticks(np.arange(preds.shape[1]), np.arange(preds.shape[1]))
	plt.rcParams.update({'font.size': 128})
	plt.yticks(np.arange(preds.shape[0]), rec_cols)
	plt.colorbar(shrink=.25, pad=.01)
	plt.xlabel('sample')
	plt.title('TruePositives  |  TPR: ' + str(np.round(100*TPR)) + '%')
	plt.grid()

	plt.subplot(222)
	plt.imshow(false_positives, vmin=0, vmax=1, cmap='magma', aspect=np.round(preds.shape[1]/preds.shape[0]))
	plt.rcParams.update({'font.size': 32})
	plt.xticks(np.arange(false_positives.shape[1]), np.arange(preds.shape[1]))
	plt.rcParams.update({'font.size': 128})
	plt.yticks(np.arange(false_positives.shape[0]), rec_cols)
	plt.colorbar(shrink=.25, pad=.01)
	plt.xlabel('sample')
	plt.title('FalsePositives  |  FPR: ' + str(np.round(100*FPR)) + '%')
	plt.grid()

	plt.subplot(223)
	plt.imshow(false_negatives, vmin=0, vmax=1, cmap='magma', aspect=np.round(preds.shape[1]/preds.shape[0]))
	plt.rcParams.update({'font.size': 32})
	plt.xticks(np.arange(false_negatives.shape[1]), np.arange(preds.shape[1]))
	plt.rcParams.update({'font.size': 128})
	plt.yticks(np.arange(false_negatives.shape[0]), rec_cols)
	plt.colorbar(shrink=.25, pad=.01)
	plt.xlabel('sample')
	plt.title('FalseNegatives  |  FNR: ' + str(np.round(100*FNR)) + '%')
	plt.grid()

	plt.subplot(224)
	plt.imshow(true_negatives, vmin=0, vmax=1, cmap='magma', aspect=np.round(preds.shape[1]/preds.shape[0]))
	plt.rcParams.update({'font.size': 32})
	plt.xticks(np.arange(true_negatives.shape[1]), np.arange(preds.shape[1]))
	plt.rcParams.update({'font.size': 128})
	plt.yticks(np.arange(true_negatives.shape[0]), rec_cols)
	plt.colorbar(shrink=.25, pad=.01)
	plt.xlabel('sample')
	plt.title('TrueNegatives  |  TNR: ' + str(np.round(100*TNR)) + '%')
	plt.grid()

	plt.savefig('preds_' + str(label) + '.pdf')
	plt.close()




filename = 'sebo.csv'
dummy = []
with open(filename) as csv_file:
	csv_reader = csv.reader(csv_file)
	line_count = 0
	for row in csv_reader:
		dummy.append(row)

feature_cols = [
		'sample_id',
		'age',
		'weight',
		'gender',
		'pregnancy_0',
		'pregnancy_1',
		'pregnancy_2',
		'pregnancy_3',
		'heavy_periods',
		'A_0',
		'A_1',
		'A_2',
		'A_3',
		'A_4',
		'A_5',
		'A_6',
		'A_7',
		'A_8',
		'A_9',
		'A_10',
		'A_11',
		'any_meds',
		'M_0',
		'M_1',
		'M_2',
		'M_3',
		'M_4',
		'M_5',
		'M_6',
		'M_7',
		'M_8',
		'M_9',
		'M_10',
		'M_11',
		'diet_0',
		'diet_1',
		'diet_2',
		'diet_3',
		'diet_4',
		'meat_freq',
		'fish_freq',
		'milk_freq',
		'fruit_freq',
		'booze_freq',
		'any_supplements',
		'num_supplements',
		'ayurveda',
		'smoking',
		'sun',
		'sport',
		'sleep',
		'Goal_0',
		'Goal_1',
		'Goal_2',
		'Goal_3',
		'Goal_4',
		'Goal_5',
		'Goal_6',
		'Goal_7',
		'stress_level',
		'stress_0',
		'stress_1',
		'stress_2',
		'stress_3',
		'stress_4',
		'burned_out',
		'heart_problems',
		'heart_0',
		'heart_1',
		'heart_2',
		'heart_3',
		'concentration',
		'memory_problems',
		'joint_problems',
		'bone_breaks',
		'often_tired',
		'challenges_ahead',
		'shit_freq',
		'shit_issue_0',
		'shit_issue_1',
		'shit_issue_2',
		'shit_issue_3',
		'shit_issue_4',
		'shit_issue_5',
		'cold',
		'public_places',
		'often_ill',
		'heavy_training',
		'Skin_0',
		'Skin_1',
		'Skin_2',
		'Skin_3',
		'Skin_4',
		'Skin_5'
		]

rec_cols = np.asarray(dummy)[2,43:]

for row in range(np.asarray(dummy).shape[0]):

	if row > first_data_line:

		data_line = np.asarray(dummy)[row,:]


		feat_dummy_row = []
		rec_dummy_row = []
		for col in range(len(data_line)):


			dummy_point = data_line[col]


			if col == 0: feat_dummy_row.append(int(dummy_point))

			if col == 1: 
				if int(dummy_point) < 25: feat_dummy_row.append(0)
				elif int(dummy_point) < 45: feat_dummy_row.append(1)
				else: feat_dummy_row.append(2)

			if col == 2: 
				if int(dummy_point) < 60: feat_dummy_row.append(0)
				elif int(dummy_point) < 75: feat_dummy_row.append(1)
				elif int(dummy_point) < 90: feat_dummy_row.append(2)
				else: feat_dummy_row.append(3)

			if col == 3:
				if dummy_point == 'weiblich': feat_dummy_row.append(0)
				else: feat_dummy_row.append(1)

			if col == 4:
				prego_dummy = []
				if dummy_point == '': 
					for j in range(4): feat_dummy_row.append(0)
				else:
					for j in range(len(dummy_point)): 
						if dummy_point[j] in ['0', '1']: prego_dummy.append(int(dummy_point[j]))
					for j in range(len(prego_dummy)): feat_dummy_row.append(prego_dummy[j])

			# if col == 4:
			# 	if dummy_point == '': feat_dummy_row.append(0)
			# 	if dummy_point == '0, 0, 0, 1': feat_dummy_row.append(0)
			# 	if dummy_point == '1, 0, 0, 0': feat_dummy_row.append(1)
			# 	if dummy_point == '0, 1, 0, 0': feat_dummy_row.append(2)
			# 	if dummy_point == '0, 0, 1, 0': feat_dummy_row.append(3)

			if col in [5, 7, 14, 15, 19, 26, 28, 29, 30, 31, 32, 33, 36, 37, 38, 39, 41]:
				if dummy_point == '': feat_dummy_row.append(0)
				if dummy_point == 'nein': feat_dummy_row.append(0)
				if dummy_point == 'ja': feat_dummy_row.append(1)

			if col == 6:
				allergies_dummy = []
				for j in range(len(dummy_point)): 
					if dummy_point[j] in ['0', '1']: allergies_dummy.append(int(dummy_point[j]))
				for j in range(len(allergies_dummy)): feat_dummy_row.append(allergies_dummy[j])

			if col == 8:
				meds_dummy = []
				if dummy_point == '': 
					for j in range(12): feat_dummy_row.append(0)
				else:
					for j in range(len(dummy_point)): 
						if dummy_point[j] in ['0', '1']: meds_dummy.append(int(dummy_point[j]))
					for j in range(len(meds_dummy)): feat_dummy_row.append(meds_dummy[j])

			# if col == 9:
			# 	if dummy_point == '': feat_dummy_row.append(0)
			# 	elif dummy_point == '0, 0, 0, 0, 1': feat_dummy_row.append(0)
			# 	elif dummy_point == '0, 0, 0, 1, 0': feat_dummy_row.append(1)
			# 	elif dummy_point == '0, 0, 1, 0, 0': feat_dummy_row.append(2)
			# 	elif dummy_point == '0, 1, 0, 0, 0': feat_dummy_row.append(3)
			# 	elif dummy_point == '1, 0, 0, 0, 0': feat_dummy_row.append(4)
			# 	else: feat_dummy_row.append(0)

			if col in [10, 11, 12, 13, 16, 17, 18, 20, 23, 34]:
				if dummy_point == '': feat_dummy_row.append(0)
				elif dummy_point == '0, 0, 1': feat_dummy_row.append(0)
				elif dummy_point == '0, 1, 0': feat_dummy_row.append(1)
				elif dummy_point == '1, 0, 0': feat_dummy_row.append(2)
				else: feat_dummy_row.append(0)

			if col == 21:
				if dummy_point == '>7': feat_dummy_row.append(1)
				else: feat_dummy_row.append(0)

			if col == 22:
				goal_dummy = []
				if dummy_point == '': 
					for j in range(8): feat_dummy_row.append(0)
				else:
					for j in range(len(dummy_point)): 
						if dummy_point[j] in ['0', '1']: goal_dummy.append(int(dummy_point[j]))
					for j in range(len(goal_dummy)): feat_dummy_row.append(goal_dummy[j])

			if col in [9, 24, 40]:
				stress_dummy = []
				if dummy_point == '': 
					for j in range(5): feat_dummy_row.append(0)
				else:
					for j in range(len(dummy_point)): 
						if dummy_point[j] in ['0', '1']: stress_dummy.append(int(dummy_point[j]))
					for j in range(len(stress_dummy)): feat_dummy_row.append(stress_dummy[j])

			if col == 25:
				if dummy_point == '': feat_dummy_row.append(0)
				if dummy_point == '0, 0, 1': feat_dummy_row.append(1)
				if dummy_point == '0, 1, 0': feat_dummy_row.append(0)
				if dummy_point == '1, 0, 0': feat_dummy_row.append(2)

			if col == 27:
				heart_dummy = []
				if dummy_point == '': 
					for j in range(4): feat_dummy_row.append(0)
				else:
					for j in range(len(dummy_point)): 
						if dummy_point[j] in ['0', '1']: heart_dummy.append(int(dummy_point[j]))
					for j in range(len(heart_dummy)): feat_dummy_row.append(heart_dummy[j])

			if col == 35:
				shit_dummy = []
				if dummy_point == '': 
					for j in range(6): feat_dummy_row.append(0)
				else:
					for j in range(len(dummy_point)): 
						if dummy_point[j] in ['0', '1']: shit_dummy.append(int(dummy_point[j]))
					for j in range(len(shit_dummy)): feat_dummy_row.append(shit_dummy[j])

			if col > 42:
				if dummy_point == 'ja': rec_dummy_row.append(1)
				else: rec_dummy_row.append(0)


		if len(feat_dummy_row) == 94:
			if row == first_data_line+1: 
				feature_matrix = np.reshape(np.asarray(feat_dummy_row), [1,-1])
				rec_matrix = np.reshape(np.asarray(rec_dummy_row), [1,-1])
			else: 
				feature_matrix = np.concatenate((feature_matrix, np.reshape(np.asarray(feat_dummy_row), [1,-1])), axis=0)
				rec_matrix = np.concatenate((rec_matrix, np.reshape(np.asarray(rec_dummy_row), [1,-1])), axis=0)


for _ in range(4): print()
print('total feature matrix loaded         |  shape: ', feature_matrix.shape)
print('total recommendation matrix loaded  |  shape: ', rec_matrix.shape)
for _ in range(4): print()

full_data = np.concatenate((feature_matrix[:,1:], rec_matrix), axis=1)
full_data = normalize(full_data, axis=0)
full_data = np.asarray(full_data)
full_labels_names = np.append(feature_cols[1:], rec_cols)

cov_mat_full = np.cov(np.transpose(full_data))
cov_mat_full /= np.amax(cov_mat_full)
cov_mat_full *= 2

if covariance_matrix_full:
		plt.figure(figsize=(200,200))
		plt.imshow(cov_mat_full, vmin=-1, vmax=1, cmap='PiYG')
		# plt.imshow(cov_mat_full, cmap='PiYG')
		plt.xticks(np.arange(cov_mat_full.shape[1]), full_labels_names, rotation=90)
		plt.yticks(np.arange(cov_mat_full.shape[0]), full_labels_names)
		plt.colorbar(shrink=.25, pad=.01)
		plt.title('covariance matrix [full]')
		plt.grid()
		plt.savefig('cov_mat_full.pdf')
		plt.close()

		plt.figure(figsize=(200,200))
		plt.imshow(full_data, cmap='PiYG')
		plt.colorbar(shrink=.25, pad=.01)
		plt.grid()
		plt.savefig('raw_data.pdf')
		plt.close()


cov_mat = np.cov(np.transpose(full_data))[len(feature_cols[1:]):,:len(feature_cols[1:])]
cov_mat /= np.amax(cov_mat)
cov_mat *= 2

if covariance_matrix:
		plt.figure(figsize=(200,200))
		plt.imshow(cov_mat, vmin=-1, vmax=1, cmap='PiYG')
		plt.xticks(np.arange(cov_mat.shape[1]), feature_cols[1:], rotation=90)
		plt.yticks(np.arange(cov_mat.shape[0]), rec_cols)
		plt.colorbar(shrink=.25, pad=.01)
		plt.title('covariance matrix')
		plt.grid()
		plt.savefig('cov_mat.pdf')
		plt.close()


x_data = normalize(feature_matrix[:,1:], axis=0)


random_sample_array = np.random.choice(x_data.shape[0], x_data.shape[0], replace=False)
train_ids = random_sample_array[:int((1. - test_size) * x_data.shape[0])]
test_ids = random_sample_array[int((1. - test_size) * x_data.shape[0]):]

x_train, x_test = x_data[train_ids,:], x_data[test_ids,:]
y_train, y_test = rec_matrix[train_ids,:], rec_matrix[test_ids,:]



skl_model = MLPClassifier(
						hidden_layer_sizes=(64, 128),
						max_iter=2000
						).fit(x_data, rec_matrix)
preds = np.transpose(np.asarray(skl_model.predict_proba(x_data)))
plot_it(preds, rec_matrix, 'MLPClassifier_full')
ensemble_preds = preds.copy()

skl_model = ExtraTreesClassifier().fit(x_data, rec_matrix)
preds = np.asarray(skl_model.predict_proba(x_data))[:,:,1]
plot_it(preds, rec_matrix, 'ExtraTrees_full')
ensemble_preds += preds

skl_model = RandomForestClassifier().fit(x_data, rec_matrix)
preds = np.asarray(skl_model.predict_proba(x_data))[:,:,1]
plot_it(preds, rec_matrix, 'RandomForest_full')
ensemble_preds += preds

ensemble_preds /= np.amax(ensemble_preds)
plot_it(ensemble_preds, rec_matrix, 'Ensemble_full')



skl_model = MLPClassifier(
						hidden_layer_sizes=(64, 128),
						max_iter=2000
						).fit(x_train, y_train)
preds = np.transpose(np.asarray(skl_model.predict_proba(x_test)))
plot_it(preds, y_test, 'MLPClassifier')
ensemble_preds = preds.copy()

skl_model = ExtraTreesClassifier().fit(x_train, y_train)
preds = np.asarray(skl_model.predict_proba(x_test))[:,:,1]
plot_it(preds, y_test, 'ExtraTrees')
ensemble_preds += preds

skl_model = RandomForestClassifier().fit(x_train, y_train)
preds = np.asarray(skl_model.predict_proba(x_test))[:,:,1]
plot_it(preds, y_test, 'RandomForest')
ensemble_preds += preds

ensemble_preds /= np.amax(ensemble_preds)
plot_it(ensemble_preds, y_test, 'Ensemble')


if DecisionTree:
	skl_model = tree.DecisionTreeClassifier().fit(x_data, rec_matrix)


	dot_data = tree.export_graphviz(
								skl_model, 
								out_file=None, 
	                     		feature_names=feature_cols[1:],  
	                     		class_names=rec_cols,  
	                     		filled=True, 
	                     		rounded=True,  
	                     		special_characters=True,
	                     		rotate=False,
	                     		proportion=True)  
	graph = graphviz.Source(dot_data)  
	graph.render("made_for_tree")  


