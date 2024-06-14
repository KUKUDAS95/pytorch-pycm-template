import traceback
try:
    from .metric_custom import *
except:
    print('Please check if "model/metric_custom.py" exists, and if it does, add its path to sys.path.')
    print('If necessary, please create and write the file. If not necessary, please comment out this section.')
    print(traceback.format_exc())

from pycm import ConfusionMatrix as pycmCM
from base import base_metric, base_class_metric

""" All metrics use the one-vs-rest strategy. """


""" Accuracy """
def ACC(confusion_obj:pycmCM, classes=None, positive_class_idx=None):
    return base_metric('ACC', confusion_obj, classes, positive_class_idx)
def ACC_class(confusion_obj:pycmCM, classes=None):
    return base_class_metric('ACC', confusion_obj, classes)

""" Sensitivity, hit rate, recall, or true positive rate """
def TPR(confusion_obj:pycmCM, classes=None, positive_class_idx=None):
    return base_metric('TPR', confusion_obj, classes, positive_class_idx)
def TPR_class(confusion_obj:pycmCM, classes=None):
    return base_class_metric('TPR', confusion_obj, classes)

""" Specificity or true negative rate """
def TNR(confusion_obj:pycmCM, classes=None, positive_class_idx=None):
    return base_metric('TNR', confusion_obj, classes, positive_class_idx)
def TNR_class(confusion_obj:pycmCM, classes=None):
    return base_class_metric('TNR', confusion_obj, classes)
    
""" F1 Score """
def F1(confusion_obj:pycmCM, classes=None, positive_class_idx=None):
    return base_metric('F1', confusion_obj, classes, positive_class_idx)
def F1_class(confusion_obj:pycmCM, classes=None):
    return base_class_metric('F1', confusion_obj, classes)
    
""" PPV, Precision or positive predictive value """
def precision(confusion_obj:pycmCM, classes=None, positive_class_idx=None):
    return base_metric('PPV', confusion_obj, classes, positive_class_idx)
def precision_class(confusion_obj:pycmCM, classes=None):
    return base_class_metric('PPV', confusion_obj, classes)
    
""" AUC (Area under the ROC curve) """
def AUC(confusion_obj:pycmCM, classes=None, positive_class_idx=None):
    if positive_class_idx is None: raise ValueError('AUC requires positive_class_idx.')
    return base_metric('AUC', confusion_obj, classes, positive_class_idx)