from sklearn.metrics import RocCurveDisplay, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn import metrics
from pycm import ROCCurve

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from utils import get_color_cycle
import numpy as np
from copy import deepcopy
import warnings

from pycm import ROCCurve, ConfusionMatrix
import seaborn as sns #minimum version 0.13.0
import pandas as pd
from scipy import interpolate
import scipy.stats as st
import matplotlib.patches as mpatches
import scipy

""" 
Curve metric (i.g., ROC, PV)
"""

""" 1. Drawing a typical ROC curve """
def check_onehot_label(item, classes):
    item_class = np.unique(np.array(item), return_counts=True)[0]
    if type(item) == int: return False
    elif len(item_class) != len(classes): return False #print('class num')
    elif 0 not in item_class and 1 not in item_class: return False #print(item_class)
    else: return True

def onehot_encoding(label, classes):
    if type(classes) == np.ndarray: classes = classes.tolist() # for FutureWarning by numpy
    item = label[0]
    if not check_onehot_label(item, classes): 
        if item not in classes: classes = np.array([idx for idx in range(len(classes))])   
        label, classes = np.array(label), np.array(classes)
        if len(classes.shape)==1: classes = classes.reshape((-1, 1))
        if len(label.shape)==1: label = label.reshape((-1, 1 if type(item) not in [list, np.ndarray] else len(item)))
        oh = OneHotEncoder()
        oh.fit(classes)
        label_onehot = oh.transform(label).toarray()
    else: label_onehot = np.array(label)
    return label_onehot


def ROC(label, prob, classes:list, specific_class_idx=None):
    fpr, tpr, roc_auc = dict(), dict(), dict()
    thresholds, best_threshold_idx = dict(), dict()
    
    label_onehot = onehot_encoding(label, classes)
    prob = np.array(prob)
    
    # STEP 1. class score
    for i in range(len(classes)):
        fpr[i], tpr[i], thresholds[i] = metrics.roc_curve(label_onehot[:, i], prob[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        tpr_fpr = tpr[i] - fpr[i]
        best_threshold_idx[i] = np.argmax(tpr_fpr)
        
    # STEP 2. macro score
    fpr_grid = np.linspace(0.0, 1.0, 1000)    
    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)    
    for i in range(len(classes)):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
    # Average it and compute AUC
    mean_tpr /= len(classes)
    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
    best_threshold_idx["macro"] = np.argmax(tpr["macro"] - fpr["macro"])
    
    # STEP 3. micro score
    fpr["micro"], tpr["micro"], thresholds["micro"] = metrics.roc_curve(label_onehot.ravel(), prob.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    best_threshold_idx["micro"] = np.argmax(tpr["micro"] - fpr["micro"])
    
    # STEP 4. plot
    fig, ax = ROC_class(label, prob, classes)    
    ax.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=2,
    )    
    ax.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=2,
    ) 
    
    # To find the best point, uncomment the relevant section and use it.
    # marker_list = cycle([k for k in Line2D.markers.keys() if k not in ['None', 'none', ' ', '']])
    # for (key, best_idx), marker, color in zip(best_threshold_idx.items(), marker_list, get_color_cycle()):
    #     best_sen = tpr[key][best_idx]
    #     best_str = f'sensitivity = {best_sen:.3f}'
    #     if key not in ['macro', 'micro']:
    #         best_spec = 1-fpr[key][best_idx]
    #         best_str = f'Best Threshold of {classes[key]} | {best_str}, specificity = {best_spec:.3f} Threshold={thresholds[key][best_idx]:.3f}'
    #     else:
    #         best_spec = fpr[key][best_idx]
    #         best_str = f'Best Threshold of {key} | {best_str}, specificity = {best_spec:.3f}'
    #     ax.scatter(best_spec, best_sen, marker=marker, s=100, color=color, label=best_str)
    
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1.02), ncol=1) # ,mode='expand', borderaxespad=0.05
    fig.tight_layout(rect=[0, 0, 1, 1])
    return ax.figure # close_all_plots()


def ROC_class(label, prob, classes:list, specific_class_idx=None):
    label_onehot = onehot_encoding(label, classes)       
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot([0,1],[0,1], label='y=x', color='lightgray', linestyle="--")
    for class_id, color in zip(range(len(classes)), get_color_cycle()):
        RocCurveDisplay.from_predictions(
            label_onehot[:, class_id],
            prob[:, class_id],
            name=f"ROC curve of {classes[class_id]}",
            linewidth=1.5,
            color=color,
            ax=ax,
        )    
    _ = ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curve",
    )
    return fig, ax     # close_all_plots()
""" 1. Drawing a typical ROC curve """

""" 2. Drawing a Fixed specificity ROC curve """
def FixedNegativeROC(label, prob, classes:list, negative_class_idx:int=0):
    # Setting up for use with `pycm` 
    if type(label[0]) in [list, np.ndarray]: 
        if type(label[0]) == np.ndarray: label = label.tolist()
        label = [a.index(1.) for a in label]
    elif type(label[0]) == 'str': label = [self.classes.index(a) for a in label]
    if len(prob[0]) != len(classes): raise ValueError('Need probability values for each class.')
    actual_prob = [p[a] for p, a in zip(prob, label)]
    crv = ROCCurve(actual_vector=np.array(label), probs=np.array(prob), classes=np.unique(label).tolist())
    
    # Setting up for plot
    label_fontsize = 11
    title_fontsize, title_pad = 14, 10
    pos_classes = {idx:name for idx, name in enumerate(classes) if idx != negative_class_idx}
    
    # Show
    roc_plot = crv.plot(classes=pos_classes.keys())
    roc_plot.set_ylabel('Sensitivity', fontsize=label_fontsize)
    roc_plot.set_xlabel(f'1 - Specificity\n(Negative Class is {classes[negative_class_idx]})', fontsize=label_fontsize)
    roc_plot.figure.suptitle('')
    roc_plot.set_title('ROC Curve', fontsize=title_fontsize, pad=title_pad)
    new_legend = []
    for l in roc_plot.legend().texts:
        class_idx = int(l.get_text())
        new_legend.append(f'{classes[class_idx]} (Area = {crv.area()[class_idx]:.3f})')
    new_legend.append('y = x')
    roc_plot.legend(labels=new_legend, loc='upper left', bbox_to_anchor=(1, 1.02), ncol=1)
    roc_plot.figure.tight_layout(rect=[0, 0, 1, 1])
    return roc_plot.figure # close_all_plots()
""" 2. Drawing a Fixed specificity ROC curve """

"""Implement one vs one ROC curve, same performance as Sklearn"""
def Multi_ROC_OvO(label, prob, classes:list, specific_class_idx=None, specificity_thr=0.05):
    #Reference: https://github.com/vinyluis/Articles/blob/main/ROC%20Curve%20and%20ROC%20AUC/ROC%20Curve%20-%20Multiclass.ipynb
    
    if np.array(label).ndim >1:
        label = one_hot_to_label_encoding(label)
    
    #based label encoding
    label2str = np.array([classes[i] for i in label], dtype=object) #y_perd, y_proba is prob ->["normal", "cancer", "cancer", "normal"] using label, y_test
    pred2str = np.array([classes[i] for i in np.argmax(prob, axis=1)], dtype=object) #y_perd, y_proba is prob ->["normal", "cancer", "cancer", "normal"] using prob, y_pred
    
    #Compares each possible combination of the classes, two at a time
    classes_combinations = []
    class_list = list(classes)

    for i in class_list:
        classes_list_remove = class_list.copy()
        classes_list_remove.remove(i)
        for j in classes_list_remove:
            classes_combinations.append([i, j])
    
    subplots_size = 4
    plot_type_num = 2 #hist, roc curve  

    plt.figure(figsize = (subplots_size*(len(classes)-1), subplots_size*plot_type_num*len(classes)))
    
    bins = [i/20 for i in range(20)] + [1]
    roc_auc_ovo = {}

    prob = np.array(prob)
    
    # Make each subplot coordinate
    def hist_roc_coordinate(A):
        rows = A * 2
        cols = A - 1
        array = [[0] * cols for _ in range(rows)]
        
        hist_coords = []
        roc_coords = []
        
        value = 1
        
        for i in range(rows):
            for j in range(cols):
                array[i][j] = value
                value += 1
        
        for idx, row in enumerate(array):
            if idx % 2 == 0:
                hist_coords.extend(row)
            else:
                roc_coords.extend(row)
        
        return  hist_coords, roc_coords
    
    hist_coordinate, roc_coordinate = hist_roc_coordinate(len(classes))
    
    for i in range(len(classes_combinations)):
        # Gets the class
        comb = classes_combinations[i]
        c1 = comb[0]
        c2 = comb[1]
        c1_index = class_list.index(c1)
        title = c1 + " vs " +c2
        
        # Prepares an auxiliar dataframe to help with the plots
        df_aux = pd.DataFrame()
        df_aux['class'] = label2str
        df_aux['prob'] = prob[:, c1_index]
        
        # Slices only the subset with both classes
        df_aux = df_aux[(df_aux['class'] == c1) | (df_aux['class'] == c2)]
        df_aux['class'] = [1 if y == c1 else 0 for y in df_aux['class']]
        df_aux = df_aux.reset_index(drop = True)
        
        # Plots the probability distribution for the class and the rest
        ax = plt.subplot(plot_type_num*len(class_list), len(class_list)-1, hist_coordinate[i])
        sns.histplot(x = "prob", data = df_aux, hue = 'class', color = 'b', ax = ax, bins = bins)
        ax.set_title(title)
        ax.legend([f"Class 1: {c1}", f"Class 0: {c2}"])
        ax.set_xlabel(f"P(x = {c1})")
        ax.axis(ymin=0,ymax=100)
        
        # Calculates the ROC Coordinates and plots the ROC Curves
        ax_bottom = plt.subplot(plot_type_num*len(class_list), len(class_list)-1, roc_coordinate[i])
        tpr, fpr = get_all_roc_coordinates(df_aux['class'], df_aux['prob'])
        
        #specificity=0.95=>fpr=0.05, calculate sensitivity
        x = fpr
        y = tpr
        liner_func = interpolate.interp1d(x, y)
        
        tnr95tnr = liner_func(specificity_thr)
        
        ax_bottom.scatter(specificity_thr, tnr95tnr)
        
        #Draw ROC curve
        plot_roc_curve(tpr, fpr, scatter = False, ax = ax_bottom)
        ax_bottom.set_title("ROC Curve OvO")
        ax_bottom.legend([f"Coordinate: {specificity_thr}, {tnr95tnr:.4f}",
                        f"ROC AUC: {roc_auc_score(df_aux['class'], df_aux['prob']):.4f}"])
        
        # Calculates the ROC AUC OvO
        roc_auc_ovo[title] = roc_auc_score(df_aux['class'], df_aux['prob'])


    avg_roc_auc = 0
    i = 0

    for k in roc_auc_ovo:
        avg_roc_auc += roc_auc_ovo[k]
        i += 1

    plt.suptitle(f"average ROC AUC OvR: {avg_roc_auc/i:.4f}")     
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    return ax.figure


"""Implement one vs rest ROC curve, same performance as Sklearn"""
def Multi_ROC_OvR(label, prob, classes:list, specific_class_idx=None, specificity_thr=0.05):
    #Reference: https://github.com/vinyluis/Articles/blob/main/ROC%20Curve%20and%20ROC%20AUC/ROC%20Curve%20-%20Multiclass.ipynb
    
    if np.array(label).ndim >1:
        label = one_hot_to_label_encoding(label)
    
    #based label encoding
    label2str = np.array([classes[i] for i in label], dtype=object) #y_perd, y_proba is prob ->["normal", "cancer", "cancer", "normal"] using label, y_test
    pred2str = np.array([classes[i] for i in np.argmax(prob, axis=1)], dtype=object) #y_perd, y_proba is prob ->["normal", "cancer", "cancer", "normal"] using prob, y_pred
    
    subplots_size = 4
    plot_type_num = 2 #hist, roc curve
    
    plt.figure(figsize = (subplots_size*len(classes), subplots_size*plot_type_num))
    
    bins = [i/20 for i in range(20)] + [1]
    roc_auc_ovr = {}
    
    prob = np.array(prob)
    
    for i in range(len(classes)):
        # Gets the class
        c = classes[i] #one이 되는 클래스
        
        # Prepares an auxiliar dataframe to help with the plots
        df_aux = pd.DataFrame()
        df_aux['class'] = [1 if y == c else 0 for y in label2str]
        df_aux['prob'] = prob[:, i]
        df_aux = df_aux.reset_index(drop = True)
        
        # Plots the probability distribution for the class and the rest
        ax = plt.subplot(2, len(classes), i+1)

        sns.histplot(x = "prob", data = df_aux, hue = 'class', color = 'b', ax = ax, bins = bins)
        ax.set_title(c)
        ax.legend([f"Class: {c}", "Rest"])
        ax.set_xlabel(f"P(x = {c})")
        ax.axis(ymin=0,ymax=200)
        
        # Calculates the ROC Coordinates and plots the ROC Curves
        ax_bottom = plt.subplot(2, len(classes), i+1+len(classes))

        tpr, fpr = get_all_roc_coordinates(df_aux['class'], df_aux['prob'])
        
        #specificity=0.95=>fpr=0.05, calculate sensitivity
        x = fpr
        y = tpr
        liner_func = interpolate.interp1d(x, y)
        
        tnr95tnr = liner_func(specificity_thr)
        
        ax_bottom.scatter(specificity_thr, tnr95tnr)
        
        plot_roc_curve(tpr, fpr, scatter = False, ax = ax_bottom)
        ax_bottom.set_title("ROC Curve OvR")
        
        ax_bottom.legend([f"Coordinate: {specificity_thr}, {tnr95tnr:.4f}",
            f"ROC AUC: {roc_auc_score(df_aux['class'], df_aux['prob']):.4f}"], loc='lower right')
            
        # Calculates the ROC AUC OvR
        roc_auc_ovr[c] = roc_auc_score(df_aux['class'], df_aux['prob'])

    avg_roc_auc = 0

    for k in roc_auc_ovr:
        avg_roc_auc += roc_auc_ovr[k]
    
    plt.suptitle(f"average ROC AUC OvR: {avg_roc_auc/len(roc_auc_ovr):.4f}")  
    plt.tight_layout()
    
    return ax.figure
    
    
"""Implement addtional function for one vs rest / one vs one ROC curve"""
def calculate_tpr_fpr(y_real, y_pred):
    '''
    Calculates the True Positive Rate (tpr) and the True Negative Rate (fpr) based on real and predicted observations
    
    Args:
        y_real: The list or series with the real classes
        y_pred: The list or series with the predicted classes
        
    Returns:
        tpr: The True Positive Rate of the classifier
        fpr: The False Positive Rate of the classifier
    '''
    
    # Calculates the confusion matrix and recover each element
    cm = confusion_matrix(y_real, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    
    # Calculates tpr and fpr
    tpr =  TP/(TP + FN) # sensitivity - true positive rate
    fpr = 1 - TN/(TN+FP) # 1-specificity - false positive rate
    
    return tpr, fpr


def get_all_roc_coordinates(y_real, y_proba):
    '''
    Calculates all the ROC Curve coordinates (tpr and fpr) by considering each point as a threshold for the predicion of the class.
    
    Args:
        y_real: The list or series with the real classes.
        y_proba: The array with the probabilities for each class, obtained by using the `.predict_proba()` method.
        
    Returns:
        tpr_list: The list of TPRs representing each threshold.
        fpr_list: The list of FPRs representing each threshold.
    '''
    tpr_list = [0]
    fpr_list = [0]
    for i in range(len(y_proba)):
        threshold = y_proba[i]
        y_pred = y_proba >= threshold
        tpr, fpr = calculate_tpr_fpr(y_real, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list


def plot_roc_curve(tpr, fpr, scatter = True, ax = None):
    '''
    Plots the ROC Curve by using the list of coordinates (tpr and fpr).
    
    Args:
        tpr: The list of TPRs representing each coordinate.
        fpr: The list of FPRs representing each coordinate.
        scatter: When True, the points used on the calculation will be plotted with the line (default = True).
    '''
    if ax == None:
        plt.figure(figsize = (5, 5))
        ax = plt.axes()
    
    if scatter:
        sns.scatterplot(x = fpr, y = tpr, ax = ax)
    sns.lineplot(x = fpr, y = tpr, ax = ax)
    sns.lineplot(x = [0, 1], y = [0, 1], color = 'green', ax = ax)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")


# One-hot encoding to label encoding
def one_hot_to_label_encoding(one_hot_labels):
    label_labels = np.argmax(one_hot_labels, axis=1)
    return label_labels


"""Implement Confidence Interval in each class"""
"""Clinical validation of a targeted methylation-based multi-cancer early detection test using an independent validation set Fig 3.B"""
def Specificity_Sensitivity_Confidence_Interval(label, prob, classes:list, specific_class_idx=None, CI=0.95, n_sided=2):
    
    #Reference Paper: 오태호, and 박선일. "진단검사 정확도 평가지표의 신뢰구간." Journal of veterinary clinics 32.4 (2015): 319-323.
    
    if np.array(label).ndim >1:
        label = one_hot_to_label_encoding(label)
    
    #based label encoder
    label_list = label
    pred_list = list(np.argmax(prob, axis=1))
    
    cm=ConfusionMatrix(label_list, pred_list)
    
    cm2array = cm.to_array()
    
    #0 is specificity class, 1~ is sensitivity
    denominator = [np.sum(i) for idx, i in enumerate(cm2array)]
    molecule = [i[idx] for idx, i in enumerate(cm2array)]

    spe_sen_list = np.divide(molecule, denominator)
    
    #sensitivity based sorting fuction
    def sorting(spe_sen_list):
        sen_sort_idx = np.argsort(spe_sen_list[1:])
        sen_sort_idx = np.concatenate(([0], sen_sort_idx+1))
        return sen_sort_idx
    
    #sorting value
    denominator = np.array(denominator)[sorting(spe_sen_list)]
    molecule = np.array(molecule)[sorting(spe_sen_list)]
    classes = np.array(classes)[sorting(spe_sen_list)]
    spe_sen_list = spe_sen_list[sorting(spe_sen_list)]
    
    #calculate z_score, p_value
    z_score = st.norm.ppf(1-(1-CI)/n_sided)
    p_value = scipy.stats.norm.sf(abs(z_score))*n_sided

    #calculate lower, upper CI function
    def Sensiticity_CI(denominator, molecule, z_score):
        a = molecule
        c = denominator - molecule
        
        A = 2*a+z_score**2
        B = z_score * np.sqrt(((z_score**2)+(4*a*c)/(a+c)))
        C = 2*(a + c + (z_score**2))
        
        lower_limit = (A-B)/C
        upper_limit = (A+B)/C
        
        return [round(lower_limit, 2), round(upper_limit, 2)]
        
    def Specificity_CI(denominator, molecule, z_score):
        d = molecule
        b = denominator - molecule
        
        A = 2*d+z_score**2
        B = z_score * np.sqrt(((z_score**2)+(4*b*d)/(b+d)))
        C = 2*(b + d + (z_score**2))
        
        lower_limit = (A-B)/C
        upper_limit = (A+B)/C
        
        return [round(lower_limit, 2), round(upper_limit, 2)]
        
    CI_list = []
    
    for idx, i in enumerate(spe_sen_list):
        if idx == 0:
            CI_list.append(Specificity_CI(denominator[idx], molecule[idx], z_score))
        else:
            CI_list.append(Sensiticity_CI(denominator[idx], molecule[idx], z_score))

    #calculate CI mean, error
    CI_mean = []
    CI_error = []
    
    for i in CI_list:
        CI_mean.append(np.mean(i))
        CI_error.append(np.abs(i[0]-i[1])/2)

    #define barplot palette
    palette = []

    for idx,i in enumerate(spe_sen_list):
        if idx == 0:
            palette.append('#F2AA84')
        else:
            if 0<=i<=0.25:
                palette.append('#ADE3F0')
            elif 0.25<i<=0.5:
                palette.append('#00AED6')
            elif 0.5<i<=0.75:
                palette.append('#157BD0')
            else:
                palette.append('#0856C3')
    
    #modified plot name
    plot_name = [name + " ("+str(m)+"/"+str(d)+")" for name, d, m in zip(classes, denominator, molecule)]
    
    ax = plt.figure(figsize = (16, 8))
    sns.barplot(x = plot_name, y = spe_sen_list*100, edgecolor = 'black', linewidth = 1.5, palette = palette)
    
    #define palette legend
    palette = ['#F2AA84', '#ADE3F0', '#00AED6', '#157BD0', '#0856C3']
    legend_handles = [mpatches.Patch(color=color) for color in palette]
    plt.legend(legend_handles,  ['Specificity', 'Sen<25%', '25%<Sen<50%', '50%<Sen<75%', 'Sen≥75%'], loc='upper center', bbox_to_anchor=(1, 0.5))
    
    #drawing CI
    errorbar = plt.errorbar(x=plot_name, y = np.array(CI_mean)*100,  yerr=np.array(CI_error)*100, capsize=5, fmt='o', color='black')

    #write specificity and sensitivity on errorbar
    for idx, (x, y) in enumerate(zip(errorbar[1][1]._x,errorbar[1][1]._y)):
        plt.text(x, y+2, str(np.round(spe_sen_list[idx]*100, 1)) + "%", ha='center', va="baseline")
    
    CI_int = int(CI*100)
    
    plt.xticks(rotation = 55, ha='right')
    plt.ylabel(f"Specificity & Sensitivity (±{CI_int}% CI, p: {p_value}) ", fontsize=15) 
    plt.yticks((0, 50, 100),('0%', '50%', '100%'))
    plt.tight_layout()
    
    return ax.figure

