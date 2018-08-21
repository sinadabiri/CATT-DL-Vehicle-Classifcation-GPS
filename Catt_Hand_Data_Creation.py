import pickle
import pandas as pd
import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import keras
from collections import Counter

class HandCraftedDataCreation:
    def __init__(self, class_set, class_4, num_fold):
        self.class_set = class_set
        self.class_4 = class_4
        self.num_fold = num_fold

    def hand_crafted_array_data(self):
        """
        :param max_length:
        :param class_set:
        :param class_4: 1: consider class_4 (bus), otherwise remove it from list
        :return:
        """
        total_input = []
        total_label = []
        total_ids = []
        for class_ in self.class_set:
            if all([class_ == 4, not self.class_4]):
                continue
            pickle_file = '../GPS_Trajectory_Feature/GPS_Trajectory_Feature_Clean_' + str(class_) + '.pickle'
            with open(pickle_file, 'rb') as g:
                all_trip_features = pickle.load(g)

            if class_ == 5:
                all_trip_features = all_trip_features[:100000]

            total_input.append(self.create_hand_crafted_features(all_trip_features))
            total_label.append(np.full(len(all_trip_features), fill_value=self.class_set[class_]))
            total_ids.append(np.array([trip[0] for trip in all_trip_features]))

        total_input = np.vstack(total_input)
        total_label = np.hstack(total_label)
        total_ids = np.hstack(total_ids)
        return total_input, total_label, total_ids

    def create_hand_crafted_features(self, X):
        # Following lists are hand-crafted features
        Dist = []
        AV = []
        EV = []
        VV = []
        MaxV1 = []
        MaxV2 = []
        MaxV3 = []
        MaxA1 = []
        MaxA2 = []
        MaxA3 = []
        HCR = []  # Heading Change Rate
        SR = []  # Stop Rate
        VCR = []  # Velocity Change Rate
        HC = 19  # Heading rate threshold
        VS = 3.4  # Stop rate threshold
        VR = 0.26  # VCR threshold
        for trip in X:  # trip: (id, trip_feature)
            RD = trip[1][0]
            DT = trip[1][1]
            SP = trip[1][2]
            AC = trip[1][3]
            BR = trip[1][4]
            VC = trip[1][5]  # relative speed, (velocity change)

            # Basic features
            # Dist: Distance of segments
            Dist.append(np.sum(RD))
            # AV: average velocity
            AV.append(np.sum(RD) / np.sum(DT))
            # EV: expectation velocity
            EV.append(np.mean(SP))
            # VV: variance of velocity
            VV.append(np.var(SP))
            # MaxV1, MaxV2, MaxV3
            sorted_velocity = np.sort(SP)[::-1]
            MaxV1.append(sorted_velocity[0])
            MaxV2.append(sorted_velocity[1])
            MaxV3.append(sorted_velocity[2])
            # MaxA1, MaxA2, MaxA3
            sorted_acceleration = np.sort(AC)[::-1]
            MaxA1.append(sorted_acceleration[0])
            MaxA2.append(sorted_acceleration[1])
            MaxA3.append(sorted_acceleration[2])
            # Heading change rate (HCR)
            Pc = sum(1 for item in list(BR) if item > HC)
            HCR.append(Pc * 1. / np.sum(RD))
            # Stop Rate (SR)
            Ps = sum(1 for item in list(SP) if item < VS)
            SR.append(Ps * 1. / np.sum(RD))
            # Velocity Change Rate (VCR)
            Pv = sum(1 for item in list(VC) if item > VR)
            VCR.append(Pv * 1. / np.sum(RD))

        X_hand = [Dist, AV, EV, VV, MaxV1, MaxV2, MaxV3, MaxA1, MaxA2, MaxA3, HCR, SR, VCR]
        X_hand = np.array(X_hand, dtype=np.float32).T

        header = ['Distance', 'Average Velocity', 'Expectation Velocity', 'Variance of Velocity', 'MaxV1',
                  'MaxV2', 'MaxV3',
                  'MaxA1', 'MaxA2', 'MaxA3', 'Heading Rate Change', 'Stop Rate', 'Velocity Change Rate',
                  'Label']
        return X_hand

    # =====================================================================================================================

    def k_fold_stratified(self):
        x, y, ids = self.hand_crafted_array_data()
        random.seed(7)
        np.random.seed(7)
        l = np.arange(len(y))
        np.random.shuffle(l)
        x = x[l]
        y = y[l]
        ids = ids[l]

        num_class = np.unique(y).shape[0]
        kfold_index = [[] for _ in range(self.num_fold)]
        for i in range(num_class):
            label_index = np.where(y == i)[0]
            for j in range(self.num_fold):
                portion = label_index[
                          round(j * 1 / self.num_fold * len(label_index)):round((j + 1) * 1 / self.num_fold * len(label_index))]
                kfold_index[j].append(portion)

        kfold_dataset = [[] for _ in range(self.num_fold)]
        all_index = np.arange(0, len(y))
        for j in range(self.num_fold):
            test_index = np.hstack(tuple([index for index in kfold_index[j]]))
            np.random.shuffle(test_index)
            Test_X = x[test_index]
            Test_Y = y[test_index]
            Test_ids = ids[test_index]
            train_index = np.delete(all_index, test_index)
            Train_X = x[train_index]
            Train_Y = y[train_index]
            Train_ids = ids[train_index]

            kfold_dataset[j] = [Train_X, Train_Y, Test_X, Test_Y, Train_ids, Test_ids]

        with open('kfold_dataset_hand_light&mid&heavy.pickle', 'wb') as f:
            pickle.dump(kfold_dataset, f)
        filename = 'kfold_debug_dataset_hand_light&mid&heavy.pickle'
        with open(filename, 'wb') as f:
            pickle.dump([[kfold_dataset[0][0][:3000], kfold_dataset[0][1][:3000], kfold_dataset[0][2][:1000],
                          kfold_dataset[0][3][:1000], kfold_dataset[0][4][:2000], kfold_dataset[0][5][:2000]]], f)

    # End of Hand_crafted Data Creation
    # =====================================================================================================================================


class MachineLearningHandCrafted:
    # Apply ML methods to Hand_Crafted Data
    def __init__(self, kfold_dataset, ml_method, num_fold_training):
        self.kfold_dataset = kfold_dataset
        self.ml_method = ml_method
        self.num_fold_training = num_fold_training # 'All': Average over all folds, otherwise apply learning on only one fold (usually for testing)
    # ===================================
    @staticmethod
    def balance_batch_size(train_x, train_y):
        num_class = len(np.unique(train_y))
        dominant = Counter(train_y).most_common(1)[0]
        dominant_class = dominant[0]
        dominant_number = dominant[1]
        x_all_class = []
        y_all_class = []
        for i in range(num_class):
            if i == dominant_class:
                class_dominant_index = np.where(train_y == i)[0]
                x_dominant = train_x[class_dominant_index]
                y_dominant = train_y[class_dominant_index]
                x_all_class.append(x_dominant)
                y_all_class.append(y_dominant)
                continue
            # resample with replacement the non-dominant class to get equal number of samples from each class
            class_non_dominant_index = np.where(train_y == i)[0]
            shortage_len = dominant_number - len(class_non_dominant_index)
            shortage_index = np.random.choice(class_non_dominant_index, size=shortage_len, replace=True)
            class_non_dominant_index_all = np.concatenate((class_non_dominant_index, shortage_index))
            x_non_dominant = train_x[class_non_dominant_index_all]
            y_non_dominant = train_y[class_non_dominant_index_all]

            x_all_class.append(x_non_dominant)
            y_all_class.append(y_non_dominant)

        Train_X = np.vstack(x_all_class)
        Train_Y = np.hstack(y_all_class)

        return Train_X, Train_Y

    @staticmethod
    def categorical_roc_score(Test_Y_ori, pred_prob):
        num_classes = len(np.unique(Test_Y_ori))
        roc_score_all = []
        for i in range(num_classes):
            Test_Y_temp = Test_Y_ori.copy()
            Test_Y_temp[np.where(Test_Y_ori==i)[0]] = 1
            Test_Y_temp[np.where(Test_Y_ori!=i)[0]] = 0
            roc_score_all.append(roc_auc_score(Test_Y_temp, pred_prob[:, i], average=None))
        return np.mean(roc_score_all)

    def ml_fit_predict(self, Train_X, Train_Y, Test_X):
        self.ml_method.fit(Train_X, Train_Y)
        pred_prob = self.ml_method.predict_proba(Test_X)
        y_pred = np.argmax(pred_prob, axis=1)
        return y_pred, pred_prob

    def training_all_folds(self):
        test_accuracy_fold = []
        ave_roc_auc_fold = []
        ave_recall_fold = []
        std_recall_fold = []

        confusion_matrix_fold = []
        cls_report = []
        for fold in self.kfold_dataset:
            Train_X = fold[0]
            Train_Y = fold[1]
            Train_X, Train_Y = MachineLearningHandCrafted.balance_batch_size(Train_X, Train_Y)
            Test_X = fold[2]
            Test_Y_ori = fold[3]
            Train_ids = fold[4]
            Test_ids = fold[5]
            y_pred, pred_prob = self.ml_fit_predict(Train_X, Train_Y, Test_X)
            test_accuracy_fold.append(accuracy_score(Test_Y_ori, y_pred))
            ave_roc_auc_fold.append(MachineLearningHandCrafted.categorical_roc_score(Test_Y_ori, pred_prob))

            recall_all_class = recall_score(Test_Y_ori, y_pred, average=None)
            ave_recall_fold.append(np.mean(recall_all_class))
            std_recall_fold.append(np.std(recall_all_class))

            confusion_matrix_fold.append(confusion_matrix(Test_Y_ori, y_pred))
            cls_report.append(classification_report(Test_Y_ori, y_pred))
            if self.num_fold_training != 'All':
                break
        accuracy_all = np.array(test_accuracy_fold)
        mean_acc = np.mean(accuracy_all)
        std_acc = np.std(accuracy_all)

        ave_recall_all = np.array(ave_recall_fold)
        mean_ave_recall = np.mean(ave_recall_all)
        std_ave_recall = np.std(ave_recall_all)

        std_recall_all = np.array(std_recall_fold)
        mean_std_recall = np.mean(std_recall_all)
        std_std_recall = np.std(std_recall_all)

        mean_auc = np.mean(ave_roc_auc_fold)
        std_auc = np.std(ave_roc_auc_fold)

        print('ML method: ', self.ml_method)
        print('CNN Accuracy:  Mean {}, std {}'.format(mean_acc, std_acc))
        print('CNN AUC Score:: Mean {}, std {}'.format(mean_auc, std_auc))
        print('CNN Average Recall Score:: Mean {}, std {}'.format(mean_ave_recall, std_ave_recall))
        print('CNN STD Recall Score:: Mean {}, std {}'.format(mean_std_recall, std_std_recall))


        #for i in range(len(cls_report)):
            #print('Confusion Matrix for fold {} : {}'.format(i, confusion_matrix_fold[i]))
            #print('\n')
            #print(cls_report[i])
        print('\n')


if __name__ == '__main__':
    # class_set={2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1}
    # {2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1}

    hand_craft_data = HandCraftedDataCreation(num_fold=5, class_set={2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2}, class_4=1)
    hand_craft_data.k_fold_stratified()  # If hand_crafted data are already created, no need to create them again

    filename = 'kfold_dataset_hand_2&3.pickle'  # or kfold_debug_dataset_hand_2class.pickle
    with open(filename, 'rb') as f:
        kfold_dataset = pickle.load(f)
    all_ml_methods = [KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), MLPClassifier(early_stopping=True, batch_size=18)]
    all_ml_methods = [CalibratedClassifierCV(LinearSVC())]
    for ml_method in all_ml_methods:
        ml_hand_craft = MachineLearningHandCrafted(kfold_dataset=kfold_dataset, ml_method=ml_method,
                                                   num_fold_training='All')
        ml_hand_craft.training_all_folds()

a = 1