import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_1samp
import pickle
from utils.const import (SUBJECTS, LABEL_NAMES)

class FovealDecoding:
    """
    Class to perform foveal decoding analysis.
    """
    
    def __init__(self, full_dataset: dict):
        """Initialize with the full dataset."""
        self.subjects = SUBJECTS
        self.label_names = LABEL_NAMES
        self.subject = 0
        self.full_dataset = full_dataset
        self.dataset_exp = 0
        self.dataset_control = 0
        self.condition = 0
        self.data = 0
        self.runs = 0
        self.labels = 0
        self.accuracy_matrices_1v1 = {}
        self.decision_score_regressors = {}
        self.blockwise_accuracy = pd.DataFrame()
        self.accuracy_matrices_cross_condition = {}
        self.cross_nobis_matrices = {}
        self.accuracies_cross_class = {"category": [], "shape": []}
        
    def run_all_decoding(self, condition, comparison='all', print_accuracy=True):
        """Run all decoding analyses and print overall accuracies."""
        accuracies = []
        accuracies_cross = []
        for s in range(len(self.subjects)):
            self.setup_variables(s=s)
            if self.dataset_exp["data"].size == 0:
                print(f"Skipping {self.subject}")
                continue
            self.set_condition(condition=condition)
            self.run_classifier_1v1() 
            self.run_cross_condition_classifier()
            self.run_cross_class_classifier()
            
            if comparison == 'all':
                accuracies.append(np.nanmean(self.accuracy_matrices_1v1[self.subject]))
                accuracies_cross.append(np.nanmean(self.accuracy_matrices_cross_condition[self.subject]))
            elif comparison == 'shape':
                accuracies.append(np.mean([self.accuracy_matrices_1v1[self.subject][0,1], self.accuracy_matrices_1v1[self.subject][2,3]]))
                accuracies_cross.append(np.mean([self.accuracy_matrices_cross_condition[self.subject][0,1], self.accuracy_matrices_cross_condition[self.subject][2,3]]))
            elif comparison == 'category':
                accuracies.append(np.mean([self.accuracy_matrices_1v1[self.subject][0,2], self.accuracy_matrices_1v1[self.subject][1,3]]))
                accuracies_cross.append(np.mean([self.accuracy_matrices_cross_condition[self.subject][0,2], self.accuracy_matrices_cross_condition[self.subject][1,3]]))
            elif comparison == 'mixed':
                accuracies.append(np.mean([self.accuracy_matrices_1v1[self.subject][0,3], self.accuracy_matrices_1v1[self.subject][1,2]]))
                accuracies_cross.append(np.mean([self.accuracy_matrices_cross_condition[self.subject][0,3], self.accuracy_matrices_cross_condition[self.subject][1,2]]))
            
            if print_accuracy:
                self.print_average_accuracy(self.accuracy_matrices_1v1[self.subject], text=f"Average decoding accuracy for {self.subject}")
        
        accuracy = np.mean(accuracies)
        cross_accuracy = np.mean(accuracies_cross)
        acc_se = np.std(accuracies) / np.sqrt(len(accuracies))
        cross_se = np.std(accuracies_cross) / np.sqrt(len(accuracies_cross))
        
        print(f"Overall average decoding accuracy:       {accuracy} +- {acc_se}")
        print(f"Overall average cross-decoding accuracy: {cross_accuracy} +- {cross_se}")
        print(f"Overall average semantic transfer: {np.mean(self.accuracies_cross_class['category'])}")
        print(f"Overall average shape transfer: {np.mean(self.accuracies_cross_class['shape'])}")
        
        if self.decision_score_regressors:
            with open('/home/lkaemmer/repos/FovealDecoding/analysis_eye/parametric_modulation/design_matrix/control_regressors.pkl', 'wb') as file:
                pickle.dump(self.decision_score_regressors, file)
                
        if not self.blockwise_accuracy.empty:
            self.blockwise_accuracy.to_csv('/home/lkaemmer/repos/FovealDecoding/analysis_eye/main_analysis/data/blockwise_accuracy.csv', index=False)
            
        return self.accuracy_matrices_1v1, self.accuracy_matrices_cross_condition, self.cross_nobis_matrices, accuracy, cross_accuracy, acc_se, cross_se
            
    def setup_variables(self, s):
        """Set variables for the current subject."""
        self.subject = self.subjects[s]
        dataset = self.full_dataset[self.subject]
        self.white_covariance = dataset['covariance']
        
        runs = dataset["runs"]
        keep_exp = runs % 2 == 1
        keep_control = runs % 2 == 0

        self.dataset_exp = {
            "data": dataset["data"][keep_exp],
            "runs": dataset["runs"][keep_exp],
            "labels": dataset["labels"][keep_exp],
        }
        self.dataset_control = {
            "data": dataset["data"][keep_control],
            "runs": dataset["runs"][keep_control],
            "labels": dataset["labels"][keep_control],
        }
    
    def set_condition(self, condition: str):
        """Set condition-specific data (experimental or control)."""
        self.condition = condition
        if condition == "experimental":
            self.data = self.dataset_exp["data"]
            self.runs = self.dataset_exp["runs"]
            self.labels = self.dataset_exp["labels"]
        elif condition == "control":
            self.data = self.dataset_control["data"]
            self.runs = self.dataset_control["runs"]
            self.labels = self.dataset_control["labels"]
        else: 
            raise ValueError("Invalid condition specified. Choose 'experimental' or 'control'.")
        
    def average_data(self, chunk_size: int):
        """Average data in chunks for each label."""
        averaged_data_list = []
        new_labels_list = []
        
        for label in np.unique(self.labels):
            indices = np.where(self.labels == label)[0]
            selected_data = self.data[indices]
            
            for i in range(0, len(selected_data), chunk_size):
                chunk = selected_data[i:i + chunk_size]
                averaged_data = np.mean(chunk, axis=0)
                averaged_data_list.append(averaged_data)
                new_labels_list.append(label)
        
        self.data = np.array(averaged_data_list)
        self.labels = np.array(new_labels_list)
                
    def run_classifier_1v1(self):
        """Run pairwise one-versus-one classification."""
        class_pairs = combinations(np.unique(self.labels), 2)
        accuracy_matrix = np.full((4, 4), np.nan)

        for class1, class2 in class_pairs:
            indices = np.where((self.labels == class1) | (self.labels == class2))[0]
            pair_data = self.data[indices]
            pair_labels = self.labels[indices]

            clf = SVC(kernel='linear')
            scores = cross_val_score(clf, pair_data, pair_labels, cv=5, scoring='accuracy')
            accuracy = np.mean(scores)
            accuracy_matrix[class1-1, class2-1] = accuracy
            accuracy_matrix[class2-1, class1-1] = accuracy
            
        self.accuracy_matrices_1v1[self.subject] = accuracy_matrix
    
    def get_decision_scores(self):
        """Get decision scores for the classifier."""
        decision_score_regressor = [0] * len(self.labels)
        unique_labels = np.unique(self.labels)
        
        for l in unique_labels:
            rest_labels = np.delete(unique_labels, np.where(unique_labels == l))
            decision_scores = []
            
            for b in rest_labels:
                indices = np.where((self.labels == l) | (self.labels == b))[0]
                pair_data = self.data[indices]
                pair_labels = self.labels[indices]
                pair_labels = [1 if label == l else -1 for label in pair_labels]

                clf = SVC(kernel='linear')
                decision_score = cross_val_predict(clf, pair_data, pair_labels, cv=5, method='decision_function')
                decision_scores.append(decision_score[np.array(pair_labels) == 1])
                
            average_scores = np.mean(decision_scores, axis=0)
            l_indices = np.where(self.labels == l)[0]
            for i, index in enumerate(l_indices):
                decision_score_regressor[index] = average_scores[i]
                
        return decision_score_regressor
    
    def get_blockwise_accuarcy(self):
        """Compute block-wise classification accuracies."""
        blockwise_accuracy = [0] * len(self.labels)
        unique_labels = np.unique(self.labels)
        
        for l in unique_labels:
            rest_labels = np.delete(unique_labels, np.where(unique_labels == l))
            accuracies = []
            
            for b in rest_labels:
                indices = np.where((self.labels == l) | (self.labels == b))[0]
                pair_data = self.data[indices]
                pair_labels = self.labels[indices]
                pair_labels = [1 if label == l else -1 for label in pair_labels]
                
                clf = SVC(kernel='linear')
                predicted_labels = cross_val_predict(clf, pair_data, pair_labels, cv=5)
                instance_accuracies = []
                for true_label, predicted_label in zip(pair_labels, predicted_labels):
                    instance_accuracy = 1 if true_label == predicted_label else 0
                    instance_accuracies.append(instance_accuracy)
                instance_accuracies = np.array(instance_accuracies)
                accuracies.append(instance_accuracies[np.array(pair_labels) == 1])
                
            average_scores = np.mean(accuracies, axis=0)
            l_indices = np.where(self.labels == l)[0]
            for i, index in enumerate(l_indices):
                blockwise_accuracy[index] = average_scores[i]
        
        nan_count = 100 - len(blockwise_accuracy)
        blockwise_accuracy = blockwise_accuracy + [np.nan] * nan_count
        return blockwise_accuracy
         
    def run_cross_condition_classifier(self):
        """Run classifier using cross-condition data."""
        class_pairs = combinations(np.unique(self.dataset_control['labels']), 2)
        accuracy_matrix = np.full((4, 4), np.nan)
        train_data = self.dataset_exp['data']
        test_data = self.dataset_control['data']
        labels = self.dataset_exp['labels']

        for class1, class2 in class_pairs:
            indices = np.where((labels == class1) | (labels == class2))[0]
            pair_train_data = train_data[indices]
            pair_test_data = test_data[indices]
            pair_labels = labels[indices]

            clf = SVC(kernel='linear')
            clf.fit(pair_train_data, pair_labels)
            predictions = clf.predict(pair_test_data)
            accuracy = accuracy_score(pair_labels, predictions)

            accuracy_matrix[class1-1, class2-1] = accuracy
            accuracy_matrix[class2-1, class1-1] = accuracy

        self.accuracy_matrices_cross_condition[self.subject] = accuracy_matrix
        return

    def print_average_accuracy(self, matrix, text, sig=True):
        """Print the average accuracy with significance stars."""
        average_accuracy = round(np.nanmean(matrix), 4)
        sig_stars = self.get_participant_significance(matrix) if sig else ''
        print(f"{text}: {average_accuracy} {sig_stars}")
    
    def get_participant_significance(self, matrix):
        """Determine significance based on participant accuracy."""
        accuracies = []
        for i in range(matrix.shape[0]):
            accuracies.extend(matrix[i, j] for j in range(i + 1, matrix.shape[1]))
        _, p_value = ttest_1samp(accuracies, 0.5)
        
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return ''