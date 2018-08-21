import pickle
import numpy as np
import keras
import random

a = 2
print('sina')
b = 1

class DLDataCreation:
    def __init__(self, class_set, class_4, max_length, num_fold, standardization, is_osrm):
        self.class_set = class_set
        self.class_4 = class_4
        self.max_length = max_length
        self.num_fold = num_fold
        self.standardization = standardization
        self.is_osrm = is_osrm

    def make_fix_length(self):
        """
        Pad and trucate the trajectory arrays to have fixed length for all.
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
            if self.is_osrm == 'OSRM':
                pickle_file = '../GPS_Trajectory_Feature/GPS_Trajectory_Feature_Clean_' + str(class_) + '.pickle'
            if self.is_osrm == 'no_OSRM':
                pickle_file = '../GPS_Trajectory_Feature/GPS_Trajectory_Feature_no_OSRM_' + str(class_) + '.pickle'

            with open(pickle_file, 'rb') as g:
                all_trip_features = pickle.load(g)

            if class_ == 5:
                all_trip_features = all_trip_features[:100000]
            for trip in all_trip_features:
                trip_ = np.array(trip[1]).T
                if len(trip_) > self.max_length:
                    total_input.append(trip_[:self.max_length])
                    total_label.append(self.class_set[class_])
                    total_ids.append(trip[0])
                else:
                    trip_padded = np.pad(trip_, ((0, self.max_length - len(trip_)), (0, 0)), 'constant')
                    total_input.append(trip_padded)
                    total_label.append(self.class_set[class_])
                    total_ids.append(trip[0])
        return np.array(total_input), np.array(total_label), np.array(total_ids)

    # +========================================================================================================
    # Create the stratified k-fold dataset for deep learning while the min-max normalization has also been applied.

    def min_max_scaler(self, input, min, max):
        """
        Min_max scaling of each channel.
        :param input: [samples, num_legs, num_features]
        :param min:
        :param max:
        :return:
        """
        current_minmax = [(np.min(input[:, :, i]), np.max(input[:, :, i])) for i in range(input.shape[2])]
        for index, item in enumerate(current_minmax):
            input[:, :, index] = (input[:, :, index] - item[0]) / (item[1] - item[0]) * (max - min) + min
        return input, current_minmax

    def standardization_scaler(self, input):
        """
        Min_max scaling of each channel.
        :param input: [samples, num_legs, num_features]
        :param min:
        :param max:
        :return:
        """
        current_mean_std = [(np.mean(input[:, :, i]), np.std(input[:, :, i])) for i in range(input.shape[2]-3)]  # excluding one-hot vector of road type
        for index, mean_std in enumerate(current_mean_std):
            input[:, :, index] = (input[:, :, index] - mean_std[0]) / mean_std[1]
        return input, current_mean_std

    def train_val_split(self, Train_X, Train_Y_ori, Train_ids, num_class):
        np.random.seed(7)
        random.seed(7)
        val_index = []
        for i in range(num_class):
            label_index = np.where(Train_Y_ori == i)[0]
            val_index.append(np.random.choice(label_index, size=round(0.1 * len(label_index)), replace=False, p=None))
        val_index = np.hstack(tuple([label for label in val_index]))
        np.random.shuffle(val_index)
        Val_X = Train_X[val_index]
        Val_Y_ori = Train_Y_ori[val_index]
        Val_ids = Train_ids[val_index]
        Val_Y = keras.utils.to_categorical(Val_Y_ori, num_classes=num_class)
        train_index_ = np.delete(np.arange(len(Train_Y_ori)), val_index)
        Train_X = Train_X[train_index_]
        Train_Y_ori = Train_Y_ori[train_index_]
        Train_ids = Train_ids[train_index_]
        Train_Y = keras.utils.to_categorical(Train_Y_ori, num_classes=num_class)
        return Train_X, Train_Y, Val_X, Val_Y, Train_ids, Val_ids

    def k_fold_stratified(self):
        x, y, ids = self.make_fix_length()
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
            Test_Y_ori = y[test_index]
            Test_ids = ids[test_index]
            train_index = np.delete(all_index, test_index)
            Train_X = x[train_index]
            Train_Y_ori = y[train_index]
            Train_ids = ids[train_index]
            if self.standardization == 'mean_std':
                Train_X, current_mean_std = self.standardization_scaler(Train_X)
                for index, mean_std in enumerate(current_mean_std):
                    Test_X[:, :, index] = (Test_X[:, :, index] - mean_std[0]) / mean_std[1]

            # Scaling to [0, 1]
            if self.standardization == 'min_max':
                Train_X, current_minmax = self.min_max_scaler(Train_X, 0, 1)
                for index, item in enumerate(current_minmax):
                    Test_X[:, :, index] = (Test_X[:, :, index] - item[0]) / (item[1] - item[0])


            Train_X, Train_Y, Val_X, Val_Y, Train_ids, Val_ids = self.train_val_split(Train_X, Train_Y_ori, Train_ids,
                                                                                 num_class=num_class)

            # Add one channel to X array for using in CNN model
            Train_X = np.reshape(Train_X, list(Train_X.shape) + [1])
            Val_X = np.reshape(Val_X, list(Val_X.shape) + [1])
            Test_X = np.reshape(Test_X, list(Test_X.shape) + [1])

            kfold_dataset[j] = [Train_X, Train_Y, Val_X, Val_Y, Test_X, Test_Y_ori, Train_ids, Val_ids, Test_ids]

        return kfold_dataset


if __name__ == '__main__':
    # Various class_set: class_set={2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1}
    # class_set = {2: 0, 3: 1}
    # class_set = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 6, 10: 6, 11: 6, 12: 6}
    # class_set = {2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2}
    # {2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1}
    # {2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2}
    # {2: 0, 3: 1}
    deep_learning_data = DLDataCreation(class_set={2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3, 8: 3, 9: 3, 10: 3, 11: 3, 12: 3}, class_4=1, max_length=70, num_fold=5, standardization='mean_std', is_osrm='OSRM')
    kfold_dataset = deep_learning_data.k_fold_stratified()
    filename = 'kfold_dataset_DL_4cat_' + deep_learning_data.standardization + '_' + deep_learning_data.is_osrm + '_.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(kfold_dataset, f)
    filename = 'kfold_debug_dataset_4cat_' + deep_learning_data.standardization + '_' + deep_learning_data.is_osrm + '_.pickle'
    with open(filename, 'wb') as f:
        pickle.dump([[kfold_dataset[0][0][:3000], kfold_dataset[0][1][:3000], kfold_dataset[0][2][:1000],
                      kfold_dataset[0][3][:1000], kfold_dataset[0][4][:2000], kfold_dataset[0][5][:2000],
                      kfold_dataset[0][6][:3000], kfold_dataset[0][7][:1000], kfold_dataset[0][8][:2000]]], f)

    # Test the k-fold dataset
    print(len(np.where(np.argmax(kfold_dataset[4][1], axis=1) == 0)[0])/len(kfold_dataset[4][1]))
    print(len(np.where(np.argmax(kfold_dataset[4][3], axis=1) == 0)[0])/len(kfold_dataset[4][3]))
    print(len(np.where(kfold_dataset[4][5] == 0)[0])/len(kfold_dataset[4][5]))
    print('\n')

    print(len(np.where(np.argmax(kfold_dataset[4][1], axis=1) == 1)[0])/len(kfold_dataset[4][1]))
    print(len(np.where(np.argmax(kfold_dataset[4][3], axis=1) == 1)[0])/len(kfold_dataset[4][3]))
    print(len(np.where(kfold_dataset[4][5] == 1)[0])/len(kfold_dataset[4][5]))
    print('\n')

    print(len(np.where(np.argmax(kfold_dataset[4][1], axis=1) == 2)[0])/len(kfold_dataset[4][1]))
    print(len(np.where(np.argmax(kfold_dataset[4][3], axis=1) == 2)[0])/len(kfold_dataset[4][3]))
    print(len(np.where(kfold_dataset[4][5] == 2)[0])/len(kfold_dataset[4][5]))
    print('\n')