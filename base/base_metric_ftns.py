from pycm import ConfusionMatrix as pycmCM
from copy import deepcopy

def base_metric(ftns_name, confusion_obj:pycmCM, classes=None, positive_class_idx=None):
    use_confusion_obj = deepcopy(confusion_obj)
    print(use_confusion_obj,"!!!!!!!!!")

    if positive_class_idx is None:
        metric_name = f'{ftns_name}_Macro' if ftns_name != 'ACC' else 'Overall_ACC'
        score = eval(f'use_confusion_obj.{metric_name}')
    else:
        classes = list(use_confusion_obj.classes)
        score = eval(f'use_confusion_obj.{ftns_name}[classes[positive_class_idx]]')
    return score if score != 'None' else 0.
    
def base_class_metric(ftns_name, confusion_obj:pycmCM, classes=None):
    use_confusion_obj = deepcopy(confusion_obj)    
    if classes is not None and classes != list(use_confusion_obj.classes):
        return {classes[class_idx]:score for class_idx, score in enumerate(eval(f'use_confusion_obj.{ftns_name}.values()'))}
    else: return eval(f'use_confusion_obj.{ftns_name}')