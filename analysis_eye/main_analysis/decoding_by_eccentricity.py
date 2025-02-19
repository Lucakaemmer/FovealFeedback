"""
Run decoding analyses by eccentricity.
"""

from utils.prepare_data import PrepareData
from utils.foveal_decoding import FovealDecoding
import matplotlib.pyplot as plt
import pandas as pd

accuracies_exp_all = pd.DataFrame()
accuracies_control_all = pd.DataFrame()
accuracies_cross_all = pd.DataFrame()
se_exp_all = pd.DataFrame()
se_control_all = pd.DataFrame()
se_cross_all = pd.DataFrame()

vareas = [1,2,3] 
 
for v in vareas:
    acc_exp = []
    acc_control = []
    acc_cross = []
    se_exp = []
    se_control = []
    se_cross = []
    
    ecc = [1,2,3,4,5]
    for e in ecc:
        print(f"Visual area V{v} Degree {e} (Retinotopy)")
        data = PrepareData()
        full_dataset = data.prepare_all_data(vareas=[v], degrees=[e], event_related=False, exclude='online', print_mask=False,
                                            anatomy=True, retinotopy=True, frontal_control=False, eccen=False, object=False)

        print('Experimental Condition')
        decoder = FovealDecoding(full_dataset)
        accuracy_matrices, cross_condition_accuracy_matrices, cross_nobis_matrices, accuracy, cross_accuracy, acc_se, cross_se = decoder.run_all_decoding(condition="experimental", print_accuracy=False)
        
        acc_exp.append(accuracy)
        acc_cross.append(cross_accuracy)
        se_exp.append(acc_se)
        se_cross.append(cross_se)
        
        print('Control Condition')
        decoder = FovealDecoding(full_dataset)
        accuracy_matrices, cross_condition_accuracy_matrices, cross_nobis_matrices, accuracy, cross_accuracy, acc_se, cross_se = decoder.run_all_decoding(condition="control", print_accuracy=False)
        
        acc_control.append(accuracy)
        se_control.append(acc_se)
        print('') 
          
    ecc = [6,7,8,9,10]
    for e in ecc:
        print(f"Visual area V{v} Degree {e} (Benson)")
        data = PrepareData()
        full_dataset = data.prepare_all_data(vareas=[v], degrees=[e], event_related=False, exclude='online', print_mask=False,
                                            anatomy=True, retinotopy=False, frontal_control=False, eccen=True, object=False)

        print('Experimental Condition')
        decoder = FovealDecoding(full_dataset)
        accuracy_matrices, cross_condition_accuracy_matrices, cross_nobis_matrices, accuracy, cross_accuracy, acc_se, cross_se = decoder.run_all_decoding(condition="experimental", print_accuracy=False)
        
        acc_exp.append(accuracy)
        acc_cross.append(cross_accuracy)
        se_exp.append(acc_se)
        se_cross.append(cross_se)
        
        print('Control Condition')
        decoder = FovealDecoding(full_dataset)
        accuracy_matrices, cross_condition_accuracy_matrices, cross_nobis_matrices, accuracy, cross_accuracy, acc_se, cross_se = decoder.run_all_decoding(condition="control", print_accuracy=False)
        
        acc_control.append(accuracy)
        se_control.append(acc_se)
        print('') 
        
    accuracies_exp_all[v] = acc_exp
    accuracies_control_all[v] = acc_control
    accuracies_cross_all[v] = acc_cross
    se_exp_all[v] = se_exp
    se_control_all[v] = se_control
    se_cross_all[v] = se_cross


ecc = [1,2,3,4,5,6,7,8,9,10]
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 6))

ax1.errorbar(ecc, accuracies_exp_all[1], yerr=se_exp_all[1], marker='o', linestyle='-', color='b', label='V1')
ax1.errorbar(ecc, accuracies_exp_all[2], yerr=se_exp_all[2], marker='o', linestyle='-', color='r', label='V2')
ax1.errorbar(ecc, accuracies_exp_all[3], yerr=se_exp_all[3], marker='o', linestyle='-', color='y', label='V3')
ax1.set_title('Accuracy by Eccentricity - Experimental')
ax1.set_xlabel('Eccentricity')
ax1.set_ylabel('Accuracy')
ax1.set_xticks(ecc)
ax1.set_ylim(0.50, 0.60)
ax1.axvspan(5.5, max(ecc), color='lightgray', alpha=0.5)
ax1.legend()

ax2.errorbar(ecc, accuracies_control_all[1], yerr=se_control_all[1], marker='o', linestyle='-', color='b', label='V1')
ax2.errorbar(ecc, accuracies_control_all[2], yerr=se_control_all[2], marker='o', linestyle='-', color='r', label='V2')
ax2.errorbar(ecc, accuracies_control_all[3], yerr=se_control_all[3], marker='o', linestyle='-', color='y', label='V3')
ax2.set_title('Accuracy by Eccentricity - Control')
ax2.set_xlabel('Eccentricity')
ax2.set_ylabel('Accuracy')
ax2.set_xticks(ecc)
ax2.set_ylim(0.50, 0.9)
ax2.axvspan(5.5, max(ecc), color='lightgray', alpha=0.5)
ax2.legend()

ax3.errorbar(ecc, accuracies_cross_all[1], yerr=se_cross_all[1], marker='o', linestyle='-', color='b', label='V1')
ax3.errorbar(ecc, accuracies_cross_all[2], yerr=se_cross_all[2], marker='o', linestyle='-', color='r', label='V2')
ax3.errorbar(ecc, accuracies_cross_all[3], yerr=se_cross_all[3], marker='o', linestyle='-', color='y', label='V3')
ax3.set_title('Accuracy by Eccentricity - Cross-Decoding')
ax3.set_xlabel('Eccentricity')
ax3.set_ylabel('Accuracy')
ax3.set_xticks(ecc)
ax3.set_ylim(0.50, 0.70)
ax3.axvspan(5.5, max(ecc), color='lightgray', alpha=0.5)
ax3.legend()

plt.tight_layout()
plt.show()