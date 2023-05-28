import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
np.seterr(divide='ignore', invalid='ignore')

def NMS(detections, delta):
    
    # Array to put the results of the NMS
    detections_tmp = np.copy(detections)
    detections_NMS = np.zeros(detections.shape)-1

    # Loop over all classes
    for i in np.arange(detections.shape[-1]):
        # Stopping condition
        while(np.max(detections_tmp[:,i]) >= 0):

            # Get the max remaining index and value
            max_value = np.max(detections_tmp[:,i])
            max_index = np.argmax(detections_tmp[:,i])

            detections_NMS[max_index,i] = max_value

            detections_tmp[int(np.maximum(-(delta/2)+max_index,0)): int(np.minimum(max_index+int(delta/2), detections.shape[0])) ,i] = -1

    return detections_NMS

def get_metrics_class(gt_indexes, pred_indexes, pred_scores, delta):


    # If there are no groundtruths
    if len(gt_indexes) == 0:
        TP_class = 0
        FP_class = len(pred_indexes)
        FN_class = 0
        return TP_class, FP_class, FN_class

    # If there are no predictions
    if len(pred_indexes) == 0:
        TP_class = 0
        FP_class = 0
        FN_class = len(gt_indexes)
        return TP_class, FP_class, FN_class

    # Iterate over all groundtruths
    TP_class = 0
    FP_class = 0
    FN_class = 0
    remove_indexes = list()

    for gt_index in gt_indexes:
        # Get the predictions which are within the delta interval of each 
        max_score = -1
        max_index = None
        for pred_index, pred_score in zip(pred_indexes, pred_scores):
            # The two indexes are very close to each other, choose the one with the greatest score
            if abs(pred_index-gt_index) <= delta/2 and pred_score > max_score and pred_index not in remove_indexes:
                max_score = pred_score
                max_index = pred_index
        # If, for this groundtruth, no predictions could fit
        if max_index is None:
            FN_class += 1
        # If there is one good prediction
        else:
            TP_class += 1
            remove_indexes.append(max_index)
    
    FP_class = len(pred_indexes)-len(remove_indexes)
    return TP_class, FP_class, FN_class


def confusion_matrix_single_game(targets, closests, detections, delta, threshold):

    # Get all targets indexes for each class
    num_classes = targets.shape[1]

    TP = list()
    FP = list()
    FN = list()

    TP_visible = list()
    FP_visible = list()
    FN_visible = list()

    TP_unshown = list()
    FP_unshown = list()
    FN_unshown = list()

    # Iterate over all classes
    for i in np.arange(num_classes):
        gt_indexes_visible = np.where(targets[:,i]==1)[0]
        gt_indexes_unshown = np.where(targets[:,i]==-1)[0]

        pred_indexes_all = np.where(detections[:,i] >=threshold)[0]

        pred_indexes_visible = list()
        pred_indexes_unshown = list()

        for pred_index in pred_indexes_all:
            if closests[pred_index,i] == 1:
                pred_indexes_visible.append(pred_index)
            else:
                pred_indexes_unshown.append(pred_index)


        pred_scores_visible = detections[pred_indexes_visible,i]
        pred_scores_unshown = detections[pred_indexes_unshown,i]

        TP_class_visible, FP_class_visible, FN_class_visible = get_metrics_class(gt_indexes_visible, pred_indexes_visible, pred_scores_visible,delta)
        TP_class_unshown, FP_class_unshown, FN_class_unshown = get_metrics_class(gt_indexes_unshown, pred_indexes_unshown, pred_scores_unshown,delta)

        TP.append(TP_class_visible + TP_class_unshown)
        FP.append(FP_class_visible + FP_class_unshown)
        FN.append(FN_class_visible + FN_class_unshown)

        TP_visible.append(TP_class_visible)
        FP_visible.append(FP_class_visible)
        FN_visible.append(FN_class_visible)

        TP_unshown.append(TP_class_unshown)
        FP_unshown.append(FP_class_unshown)
        FN_unshown.append(FN_class_unshown)

    return TP, FP, FN, TP_visible, FP_visible, FN_visible, TP_unshown, FP_unshown, FN_unshown


def compute_confusion_matrix(targets, closests, detections, delta, threshold):

    TP = np.array([0]*targets[0].shape[1])
    FP = np.array([0]*targets[0].shape[1])
    FN = np.array([0]*targets[0].shape[1])
    TP_visible = np.array([0]*targets[0].shape[1])
    FP_visible = np.array([0]*targets[0].shape[1])
    FN_visible = np.array([0]*targets[0].shape[1])
    TP_unshown = np.array([0]*targets[0].shape[1])
    FP_unshown = np.array([0]*targets[0].shape[1])
    FN_unshown = np.array([0]*targets[0].shape[1])

    for target, closest, detection in zip(targets, closests, detections):
        TP_tmp, FP_tmp, FN_tmp, TP_tmp_visible, FP_tmp_visible, FN_tmp_visible, TP_tmp_unshown, FP_tmp_unshown, FN_tmp_unshown = confusion_matrix_single_game(target, closest, detection, delta, threshold)
        TP += np.array(TP_tmp)
        FP += np.array(FP_tmp)
        FN += np.array(FN_tmp)
        TP_visible += np.array(TP_tmp_visible)
        FP_visible += np.array(FP_tmp_visible)
        FN_visible += np.array(FN_tmp_visible)
        TP_unshown += np.array(TP_tmp_unshown)
        FP_unshown += np.array(FP_tmp_unshown)
        FN_unshown += np.array(FN_tmp_unshown)

    return TP, FP, FN, TP_visible, FP_visible, FN_visible, TP_unshown, FP_unshown, FN_unshown

def compute_precision_recall_curve(targets, closests, detections, delta, NMS_on):

    # 200 confidence thresholds between [0,1]
    thresholds = np.linspace(0,1,200)

    # Store the precision and recall points
    precision = list()
    recall = list()
    precision_visible = list()
    recall_visible = list()
    precision_unshown = list()
    recall_unshown = list()

    # Apply Non-Maxima Suppression if required
    detections_NMS = list()
    if NMS_on:
        for detection in detections:
            detections_NMS.append(NMS(detection,delta))
    else:
        detections_NMS = detections

    # Get the precision and recall for each confidence threshold
    for threshold in thresholds:
        TP, FP, FN, TP_visible, FP_visible, FN_visible, TP_unshown, FP_unshown, FN_unshown = compute_confusion_matrix(targets, closests, detections_NMS, delta, threshold)
        p = np.nan_to_num(TP/(TP+FP))
        r = np.nan_to_num(TP/(TP+FN))
        p_visible = np.nan_to_num(TP_visible/(TP_visible+FP_visible))
        r_visible = np.nan_to_num(TP_visible/(TP_visible+FN_visible))
        p_unshown = np.nan_to_num(TP_unshown/(TP_unshown+FP_unshown))
        r_unshown = np.nan_to_num(TP_unshown/(TP_unshown+FN_unshown))

        precision.append(p)
        recall.append(r)
        precision_visible.append(p_visible)
        recall_visible.append(r_visible)
        precision_unshown.append(p_unshown)
        recall_unshown.append(r_unshown)

    precision = np.array(precision)
    recall = np.array(recall)
    precision_visible = np.array(precision_visible)
    recall_visible = np.array(recall_visible)
    precision_unshown = np.array(precision_unshown)
    recall_unshown = np.array(recall_unshown)


    # Sort the points based on the recall, class per class
    for i in np.arange(precision.shape[1]):
        index_sort = np.argsort(recall[:,i])
        precision[:,i] = precision[index_sort,i]
        recall[:,i] = recall[index_sort,i]
    for i in np.arange(precision_visible.shape[1]):
        index_sort = np.argsort(recall_visible[:,i])
        precision_visible[:,i] = precision_visible[index_sort,i]
        recall_visible[:,i] = recall_visible[index_sort,i]
    for i in np.arange(precision_unshown.shape[1]):
        index_sort = np.argsort(recall_unshown[:,i])
        precision_unshown[:,i] = precision_unshown[index_sort,i]
        recall_unshown[:,i] = recall_unshown[index_sort,i]

    return precision, recall, precision_visible, recall_visible, precision_unshown, recall_unshown

def compute_mAP(precision, recall):

    # Array for storing the AP per class
    AP = np.array([0.0]*precision.shape[-1])

    # Loop for all classes
    for i in np.arange(precision.shape[-1]):

        # 11 point interpolation
        for j in np.arange(11)/10:

            index_recall = np.where(recall[:,i] >= j)[0]

            possible_value_precision = precision[index_recall,i]
            max_value_precision = 0

            if possible_value_precision.shape[0] != 0:
                max_value_precision = np.max(possible_value_precision)

            AP[i] += max_value_precision

    mAP_per_class = AP/11

    return np.mean(mAP_per_class), mAP_per_class

def delta_curve(targets, closests, detections,  framerate, savepath, NMS_on):

    mAP = list()
    mAP_per_class = list()
    mAP_visible = list()
    mAP_per_class_visible = list()
    mAP_unshown = list()
    mAP_per_class_unshown = list()

    for delta in tqdm((np.arange(12)*5 + 5)*framerate):

        precision, recall, precision_visible, recall_visible, precision_unshown, recall_unshown = compute_precision_recall_curve(targets, closests, detections, delta, NMS_on)

        tmp_mAP, tmp_mAP_per_class = compute_mAP(precision, recall)
        mAP.append(tmp_mAP)
        mAP_per_class.append(tmp_mAP_per_class)
        tmp_mAP_visible, tmp_mAP_per_class_visible = compute_mAP(precision_visible, recall_visible)
        mAP_visible.append(tmp_mAP_visible)
        mAP_per_class_visible.append(tmp_mAP_per_class_visible)
        tmp_mAP_unshown, tmp_mAP_per_class_unshown = compute_mAP(precision_unshown, recall_unshown)
        mAP_unshown.append(tmp_mAP_unshown)
        mAP_per_class_unshown.append(tmp_mAP_per_class_unshown)

    return mAP, mAP_per_class, mAP_visible, mAP_per_class_visible, mAP_unshown, mAP_per_class_unshown


def average_mAP_visibility_corrected(targets, detections, framerate=2, savepath=None, NMS_on=True):

    targets_numpy = list()
    closests_numpy = list()
    detections_numpy = list()
    
    for target, detection in zip(targets,detections):
        target_numpy = target.numpy()
        targets_numpy.append(target_numpy)
        detections_numpy.append(detection.numpy())

        closest_numpy = np.zeros(target_numpy.shape)-1
        #Get the closest action index
        for c in np.arange(target_numpy.shape[-1]):

            indexes = np.where(target_numpy[:,c] != 0)[0].tolist()
            if len(indexes) == 0 :
                continue
            indexes.insert(0,-indexes[0])
            indexes.append(2*closest_numpy.shape[0])
            for i in np.arange(len(indexes)-2)+1:
                start = max(0,(indexes[i-1]+indexes[i])//2)
                stop = min(closest_numpy.shape[0], (indexes[i]+indexes[i+1])//2)
                closest_numpy[start:stop,c] = target_numpy[indexes[i],c]
        closests_numpy.append(closest_numpy)



    mAP, mAP_per_class, mAP_visible, mAP_per_class_visible, mAP_unshown, mAP_per_class_unshown = delta_curve(targets_numpy, closests_numpy, detections_numpy, framerate, savepath, NMS_on)
    # Compute the average mAP
    integral = 0.0
    for i in np.arange(len(mAP)-1):
        integral += 5*(mAP[i]+mAP[i+1])/2
    a_mAP = integral/(5*(len(mAP)-1))

    integral_visible = 0.0
    for i in np.arange(len(mAP_visible)-1):
        integral_visible += 5*(mAP_visible[i]+mAP_visible[i+1])/2
    a_mAP_visible = integral_visible/(5*(len(mAP_visible)-1))

    integral_unshown = 0.0
    for i in np.arange(len(mAP_unshown)-1):
        integral_unshown += 5*(mAP_unshown[i]+mAP_unshown[i+1])/2
    a_mAP_unshown = integral_unshown/(5*(len(mAP_unshown)-1))

    a_mAP_per_class = list()
    for c in np.arange(len(mAP_per_class[0])):
        integral_per_class = 0.0
        for i in np.arange(len(mAP_per_class)-1):
            integral_per_class += 5*(mAP_per_class[i][c]+mAP_per_class[i+1][c])/2
        a_mAP_per_class.append(integral_per_class/(5*(len(mAP_per_class)-1)))

    a_mAP_per_class_visible = list()
    for c in np.arange(len(mAP_per_class_visible[0])):
        integral_per_class_visible = 0.0
        for i in np.arange(len(mAP_per_class_visible)-1):
            integral_per_class_visible += 5*(mAP_per_class_visible[i][c]+mAP_per_class_visible[i+1][c])/2
        a_mAP_per_class_visible.append(integral_per_class_visible/(5*(len(mAP_per_class_visible)-1)))

    a_mAP_per_class_unshown = list()
    for c in np.arange(len(mAP_per_class_unshown[0])):
        integral_per_class_unshown = 0.0
        for i in np.arange(len(mAP_per_class_unshown)-1):
            integral_per_class_unshown += 5*(mAP_per_class_unshown[i][c]+mAP_per_class_unshown[i+1][c])/2
        a_mAP_per_class_unshown.append(integral_per_class_unshown/(5*(len(mAP_per_class_unshown)-1)))

    return a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown