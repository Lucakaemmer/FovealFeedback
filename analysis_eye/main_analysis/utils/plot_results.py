import numpy as np
from itertools import product
from scipy.stats import ttest_1samp, ttest_rel
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from utils.const import (SUBJECTS, LABEL_NAMES, CHANCE_LEVEL)

class PlotResults:
    """
    Class for plotting decoding results.
    """

    def __init__(self, accuracy_matrices: dict):
        """Initialize with accuracy matrices and compute overall stats."""
        self.subjects = SUBJECTS
        self.label_names = LABEL_NAMES
        self.accuracy_matrices = accuracy_matrices
        self.average_accuracy_matrix = self._compute_average_accuracy_matrix()
        self.overall_accuracies = self._compute_overall_accuracies()
        self.p_value_matrix = self._compute_significance_matrix()
        self.overall_p_values = self._compute_overall_significance()
        self.colors = 0
        self.cmap = 0

    def _compute_average_accuracy_matrix(self):
        """Compute the average accuracy matrix across subjects."""
        return np.mean(np.stack(list(self.accuracy_matrices.values())), axis=0)
    
    def _compute_overall_accuracies(self):
        """Compute overall accuracies for different comparisons."""
        idx_1 = [(0, 1), (0, 2), (0, 3)]
        idx_2 = [(2, 3), (1, 3), (1, 2)]
        keys = "shape", "category", "mixed"
        accuracies = {}
        for i in range(len(keys)):
            accuracies[keys[i]] = np.array([
                np.mean([self.accuracy_matrices[s][idx_1[i]], self.accuracy_matrices[s][idx_2[i]]]) 
                for s in self.accuracy_matrices.keys()])  
        accuracies["all"] = np.mean([accuracies["shape"], accuracies["category"], accuracies["mixed"]], axis=0)
        return accuracies
    
    def _compute_significance_matrix(self):
        """Compute p-values for each pair of classes."""
        p_value_matrix = np.full((4, 4), np.nan)
        for i, j in product(range(4), range(4)):
            if i != j:
                accuracies = np.array([self.accuracy_matrices[s][i, j] for s in self.accuracy_matrices.keys()])
                _, p_value_matrix[i, j] = ttest_1samp(accuracies, CHANCE_LEVEL)
        return p_value_matrix
    
    def _compute_overall_significance(self):
        """Compute overall significance for each comparison."""
        p_values = {}
        keys = list(self.overall_accuracies.keys())
        for key in keys:
            _, p_values[key] = ttest_1samp(self.overall_accuracies[key], CHANCE_LEVEL)
        _, p_values["shape<mixed"] = ttest_rel(self.overall_accuracies["shape"], self.overall_accuracies["mixed"])
        _, p_values["category<mixed"] = ttest_rel(self.overall_accuracies["category"], self.overall_accuracies["mixed"])
        _, p_values["shape<category"] = ttest_rel(self.overall_accuracies["shape"], self.overall_accuracies["category"])
        return p_values

    def plot_bar_plot(self, title, max=0.65, bonferroni=False):
        """Plot bar plots with significance stars."""
        # Defining variables
        x_labels = ["all", "within-shape", "within-category", "mixed"]
        decoding_accuracies = [np.mean(self.overall_accuracies["all"]), np.mean(self.overall_accuracies["shape"]), np.mean(self.overall_accuracies["category"]), np.mean(self.overall_accuracies["mixed"])]
        standard_errors = [stats.sem(self.overall_accuracies["all"]),  stats.sem(self.overall_accuracies["shape"]), stats.sem(self.overall_accuracies["category"]), stats.sem(self.overall_accuracies["mixed"])]
        p_values = [self.overall_p_values["all"], self.overall_p_values["shape"], self.overall_p_values["category"], self.overall_p_values["mixed"]]
        p_values_comparison = [
            [1, 1, 1, 1],
            [1, 1, self.overall_p_values["shape<category"], self.overall_p_values["shape<mixed"]],
            [1, self.overall_p_values["shape<category"], 1, self.overall_p_values["category<mixed"]],
            [1, self.overall_p_values["shape<mixed"], self.overall_p_values["category<mixed"], 1]
        ]
        
        # Plotting the bars
        fig, ax = plt.subplots()
        bars = ax.bar(x_labels, decoding_accuracies, color=self.colors, yerr=standard_errors, capsize=5)

        # Adding stars over bars
        for bar, p in zip(bars, p_values):
            if stars := self.add_stars(p, bonferroni=bonferroni):
                ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height()-0.02, stars, ha='center', va='bottom', color='black', fontsize=12)

        # Adding significance stars between bars for p_values_comparison
        for i in range(len(decoding_accuracies)):
            for j in range(i + 1, len(decoding_accuracies)):
                p = p_values_comparison[i][j]
                if stars := self.add_stars(p, bonferroni=bonferroni):
                    self.add_stars_brackets(i, j, text=stars, center=np.arange(len(decoding_accuracies)), height=decoding_accuracies)

        # Customizing plot
        ax.axhline(y=0.5, color='black', linestyle='--')
        ax.set_xlabel('Comparison')
        ax.set_ylabel('Decoding Accuracy')
        ax.set_title(title)
        ax.set_xticks(range(len(decoding_accuracies)))
        ax.set_xticklabels(x_labels)
        ax.set_ylim(0.45, max)
        plt.show()
        
    def plotting_1v1(self, title, max=0.65, bonferroni=False):
        """Plot a heatmap of one-versus-one decoding accuracies."""
        # Create annotations with stars based on p-values
        annotations = np.empty_like(self.average_accuracy_matrix, dtype=object)
        for i in range(self.average_accuracy_matrix.shape[0]):
            for j in range(self.average_accuracy_matrix.shape[1]):
                accuracy = self.average_accuracy_matrix[i, j]
                star = self.add_stars(self.p_value_matrix[i, j], bonferroni=bonferroni)
                annotations[i, j] = f"{accuracy:.2f}  {star}"
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.average_accuracy_matrix, annot=annotations, fmt='', vmin=0.5, vmax=max, cmap=self.cmap, cbar_kws={'label': 'Accuracy'}, 
                    linewidths=.5, xticklabels=self.label_names, yticklabels=self.label_names)
        plt.title(title)
        plt.xlabel("Class")
        plt.ylabel("Class")
        plt.show() 
        
        
    def add_stars(self, p, bonferroni=False):
        """Return significance stars based on p-value."""
        if bonferroni:
            p = p * bonferroni
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''
        
    def add_stars_brackets(self, num1, num2, text, center, height, dh=.03, barh=.02):
        """Annotate bar plot with significance brackets."""
        lx, ly = center[num1], height[num1]
        rx, ry = center[num2], height[num2]
        ax_y0, ax_y1 = plt.gca().get_ylim()
        dh *= (ax_y1 - ax_y0)
        barh *= (ax_y1 - ax_y0)
        y = max(ly, ry) + dh
        barx = [lx, lx, rx, rx]
        bary = [y, y+barh, y+barh, y]
        mid = ((lx+rx)/2, y+barh)
        plt.plot(barx, bary, c='black')
        plt.text(*mid, text, ha='center', va='bottom')
      
    def set_colors(self, color):
        """Set plot colors and colormap based on input."""
        if color == "blue":
            self.colors = ['darkblue', 'cornflowerblue', 'cornflowerblue', 'cornflowerblue']
            self.cmap = "Blues"
        elif color == "green":
            self.colors = ['darkgreen', 'limegreen', 'limegreen', 'limegreen']
            self.cmap = "Greens"
        elif color == "red":
            self.colors = ['darkred', 'red', 'red', 'red']
            self.cmap = "Reds"
        elif color == "grey":
            self.colors = ['grey', 'lightgrey', 'lightgrey', 'lightgrey']
            self.cmap = "Greys"
        return
