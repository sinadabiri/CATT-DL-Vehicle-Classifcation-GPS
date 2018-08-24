import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, recall_score, roc_auc_score, roc_curve, average_precision_score, precision_recall_curve
import random
import pickle
import tensorflow as tf
from collections import Counter
import keras
import time
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


class DeepLearningModel:
    def __init__(self, num_filter, num_dense, num_conv_set, use_attention, attention_MLP, initializer, kernel_size, strides, activation, padding, learning_rate, epochs, batch_size, num_fold_training, val_threshold,
                 make_balance_dataset, make_balance_batch, alpha, weighted_logits, val_metric):
        self.num_filter = num_filter
        self.num_dense = num_dense
        self.num_conv_set = num_conv_set
        self.use_attention = use_attention
        self.attention_MLP = attention_MLP
        self.initializer = initializer
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.padding = padding
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_fold_training = num_fold_training
        self.val_threshold = val_threshold
        self.make_balance_dataset = make_balance_dataset
        self.make_balance_batch = make_balance_batch
        self.alpha = alpha
        self.weighted_logits = weighted_logits
        self.val_metric = val_metric

    def classifier(self, input_layer, num_class):
        conv_layer = input_layer
        for i in range(self.num_conv_set):
            with tf.variable_scope('conv_set_'+str(i), initializer=self.initializer, regularizer=tf.contrib.layers.l1_regularizer(scale=0.1)):
                for j in range(2):
                    kernel_size_ = [self.kernel_size[i]] + [conv_layer.get_shape().as_list()[2]]
                    conv_layer = tf.layers.conv2d(inputs=conv_layer, activation=self.activation, filters=self.num_filter[i],
                                                  name='_conv_'+str(j), kernel_size=kernel_size_, strides=self.strides, padding=self.padding)
                    shape = conv_layer.get_shape().as_list()
                    conv_layer = tf.reshape(conv_layer, shape=[-1, shape[1], shape[3], shape[2]])

        shape = conv_layer.get_shape().as_list()
        if not self.use_attention:
            pool_size = [shape[1], 1]
            conv_layer = tf.layers.max_pooling2d(conv_layer, pool_size=pool_size, strides=pool_size, name='pool')

        if self.use_attention:
            # Remove the channel dimension and make conv_layer ready as the input for attention model
            conv_layer = tf.reshape(conv_layer, shape=[-1] + shape[1:-1])
            conv_layer = self.attention(conv_layer)

        dense = tf.layers.flatten(conv_layer, name='flatten')
        units = int(dense.get_shape().as_list()[-1] / 4)
        with tf.variable_scope('dense_set_1_', reuse=tf.AUTO_REUSE, initializer=self.initializer, regularizer=tf.contrib.layers.l1_regularizer(scale=0.1)):
            for i in range(self.num_dense):
                dense = tf.layers.dense(dense, units, activation=self.activation, name=str(i))
                units /= 2
        dense = tf.layers.dropout(dense, 0.5, name='drop_out')
        classifier_output = tf.layers.dense(dense, num_class, name='softmax_layer')
        return classifier_output

    def attention(self, conv_layer):
        # conv_layer = [batch_size, num_convolved_legs, leg_vector_size]

        leg_vector_size = conv_layer.get_shape().as_list()[-1]  # get last dimension
        with tf.variable_scope('attention_set_', reuse=tf.AUTO_REUSE, initializer=self.initializer, regularizer=tf.contrib.layers.l1_regularizer(scale=0.1)):
            # v_attention: context vector
            v_attention = tf.get_variable("attention_1", shape=[leg_vector_size])

            # 1.one-layer MLP
            if self.attention_MLP:
                u = tf.layers.dense(conv_layer, leg_vector_size, activation=tf.nn.tanh, use_bias=True)  # [batch_size, num_convolved_legs, leg_vector_size].no-linear
            else:
                u = conv_layer
            # 2.compute weight by compute similarity of u and attention vector v

            score = tf.multiply(u, v_attention)  # [batch_size,seq_length,num_units]
            weight = tf.reduce_sum(score, axis=2, keepdims=True)  # [batch_size,seq_length,1]
            normalized_weight = tf.nn.softmax(weight)  # [batch_size, num_convolved_legs, 1]

            # 3.weight sum
            attention_representation = tf.reduce_sum(tf.multiply(conv_layer, normalized_weight), axis=1)  # [batch_size,num_units]
        return attention_representation

    def cnn_model(self, input_layer, true_label, num_class, non_dom_weight, dom_weight):
        classifier_output = self.classifier(input_layer, num_class)
        if self.weighted_logits:
            #class_weight = tf.constant([non_dom_weight, dom_weight])
            class_weight = tf.constant(self.weighted_logits)
            weighted_logits = tf.multiply(classifier_output, class_weight)  # shape [batch_size, 2]
            loss_cls = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_label, logits=weighted_logits,
                                                           name='loss_cls'))
        else:
            loss_cls = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_label, logits=classifier_output,
                                                           name='loss_cls'))
            # loss_reg = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'EasyNet'))
            # loss_cls += loss_reg

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss_cls)
        # Change grads_and_vars as you wish
        train_op = optimizer.minimize(loss_cls)

        correct_prediction = tf.equal(tf.argmax(true_label, 1), tf.argmax(classifier_output, 1))
        accuracy_cls = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return train_op, classifier_output, accuracy_cls

    def validation_metric(self, Val_X, Val_Y, classifier_output, input_layer, sess):
        prediction = []
        for i in range(len(Val_X) // 256):
            Test_X_batch = Val_X[i * 256:(i + 1) * 256]
            prediction.append(sess.run(tf.nn.softmax(classifier_output), feed_dict={input_layer: Test_X_batch}))
        Test_X_batch = Val_X[(i + 1) * 256:]
        if len(Test_X_batch) >= 1:
            prediction.append(sess.run(tf.nn.softmax(classifier_output), feed_dict={input_layer: Test_X_batch}))
        prediction = np.vstack(tuple(prediction))
        y_pred = np.argmax(prediction, axis=1)
        if self.val_metric == 'recall':
            return recall_score(np.argmax(Val_Y, axis=1), y_pred, average='macro')
        if self.val_metric == 'roc_auc':
            Val_Y = keras.utils.to_categorical(Val_Y, num_classes=len(np.unique(Val_Y)))  # one-hot vector of y
            return DeepLearningModel.categorical_roc_score(Val_Y, prediction)
        if self.val_metric == 'accuracy':
            return accuracy_score(np.argmax(Val_Y), y_pred)

    def prediction_prob(self, Test_X, classifier_output, input_layer, sess):
        prediction = []
        for i in range(len(Test_X) // 256):
            Test_X_batch = Test_X[i * 256:(i + 1) * 256]
            prediction.append(sess.run(tf.nn.softmax(classifier_output), feed_dict={input_layer: Test_X_batch}))
        Test_X_batch = Test_X[(i + 1) * 256:]
        if len(Test_X_batch) >= 1:
            prediction.append(sess.run(tf.nn.softmax(classifier_output), feed_dict={input_layer: Test_X_batch}))
        prediction = np.vstack(tuple(prediction))
        y_pred = np.argmax(prediction, axis=1)
        return y_pred, prediction

    # ===================================
    def balance_batch_size(self, train_x, train_y, num_class):
        dominant = Counter(np.argmax(train_y, axis=1)).most_common(1)[0]
        dominant_class = dominant[0]
        dominant_number = dominant[1]
        x_y_all_class = []
        for i in range(num_class):
            if i == dominant_class:
                class_dominant_index = np.where(np.argmax(train_y, axis=1) == i)[0]
                x_dominant = train_x[class_dominant_index]
                y_dominant = train_y[class_dominant_index]
                x_y_all_class.append((x_dominant, y_dominant))
                continue
            # resample with replacement the non-dominant class to get equal number of samples from each class
            class_non_dominant_index = np.where(np.argmax(train_y, axis=1) == i)[0]
            shortage_len = dominant_number - len(class_non_dominant_index)
            shortage_index = np.random.choice(class_non_dominant_index, size=shortage_len, replace=True)
            class_non_dominant_index_all = np.concatenate((class_non_dominant_index, shortage_index))
            x_non_dominant = train_x[class_non_dominant_index_all]
            y_non_dominant = train_y[class_non_dominant_index_all]

            x_y_all_class.append((x_non_dominant, y_non_dominant))

        class_batch = int(self.batch_size/num_class)
        num_batches = int(dominant_number // class_batch)
        all_x = []
        all_y = []
        for i in range(num_batches):
            x_batch = []
            y_batch = []
            for j in range(num_class):
                a_x = x_y_all_class[j][0][i * class_batch: (i + 1) * class_batch]
                x_batch.append(a_x)
                a_y = x_y_all_class[j][1][i * class_batch: (i + 1) * class_batch]
                y_batch.append(a_y)
            all_x.append((np.vstack(x_batch)))
            all_y.append((np.vstack(y_batch)))
        x_batch = []
        y_batch = []
        for j in range(num_class):
            a_x = x_y_all_class[j][0][(i + 1) * class_batch:]
            x_batch.append(a_x)
            all_x.append((np.vstack(x_batch)))
            a_y = x_y_all_class[j][1][(i + 1) * class_batch:]
            y_batch.append(a_y)
            all_y.append((np.vstack(y_batch)))

        Train_X = np.vstack(all_x)
        Train_Y = np.vstack(all_y)

        return Train_X, Train_Y

    def training(self, one_fold):
        Train_X = one_fold[0]
        Train_Y = one_fold[1]
        Val_X = one_fold[2]
        Val_Y = one_fold[3]
        Test_X = one_fold[4]
        Test_Y_ori = one_fold[5]
        num_class = Train_Y.shape[-1]
        input_size = list(Test_X.shape[1:])
        count = Counter(np.argmax(Train_Y, axis=1))
        non_dom_weight = math.log(len(Train_Y)/count[1])  # positive class which is always 1 in my settings
        dom_weight = math.log(len(Train_Y) / count[0])

        if self.make_balance_batch:
            Train_X, Train_Y = self.balance_batch_size(Train_X, Train_Y, num_class)

        Train_ids = one_fold[6]
        Val_ids = one_fold[7]
        Test_ids = one_fold[8]

        random.seed(7)
        np.random.seed(7)

        val_metric_epochs = {-2: 0, -1: 0}

        tf.reset_default_graph()
        tf.set_random_seed(1234)
        with tf.Session() as sess:
            input_layer = tf.placeholder(dtype=tf.float32, shape=[None] + input_size, name='input_layer')
            true_label = tf.placeholder(tf.float32, shape=[None, num_class], name='true_label')

            train_op, classifier_output, accuracy_cls = self.cnn_model(input_layer, true_label, num_class, non_dom_weight, dom_weight)

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=20)

            num_batches = len(Train_X) // self.batch_size
            k = 0
            grads_and_vars_all_epochs = []
            while True:
                grads_and_vars_all_iterations = []
                start_time = time.clock()
                for i in range(num_batches):
                    X_cls = Train_X[i * self.batch_size: (i + 1) * self.batch_size]
                    Y_cls = Train_Y[i * self.batch_size: (i + 1) * self.batch_size]

                    #grads_and_vars_ = sess.run(grads_and_vars, feed_dict={input_layer: X_cls, true_label: Y_cls})
                    #grads_and_vars_all_iterations.append(grads_and_vars_)

                    accuracy, _ = sess.run([accuracy_cls, train_op], feed_dict={input_layer: X_cls, true_label: Y_cls})

                    print('Epoch Num {}, Batches Num {}, Accuracy {}'.format(k, i, np.round(accuracy, 3)))

                X_cls = Train_X[(i + 1) * self.batch_size:]
                Y_cls = Train_Y[(i + 1) * self.batch_size:]
                if len(Y_cls) >= 1:
                    sess.run(train_op, feed_dict={input_layer: X_cls, true_label: Y_cls})
                print('Epoch Num {}, computation time: {}'.format(k, time.clock()-start_time))


                # gradient_over_iterations = [ite[0][0][0, 0, 0, 0] for ite in grads_and_vars_all_iterations]
                # for item in gradient_over_iterations:
                # print(item)
                # print('\n'); print('\n');  print('\n'); print('\n')

                #weight_over_iterations = [ite[0][1][0, 0, 0, 0] for ite in grads_and_vars_all_iterations]
                # for item in weight_over_iterations:
                # print(item)
                #grads_and_vars_ = sess.run(grads_and_vars, feed_dict={input_layer: X_cls, true_label: Y_cls})
                #grads_and_vars_all_epochs.append(grads_and_vars_)

                print('====================================================')
                val_metric_value = self.validation_metric(Val_X, Val_Y, classifier_output, input_layer, sess)
                print('Epoch Num {}, Val_Metric_Type {}, {}'.format
                      (k, self.val_metric, np.round(val_metric_value, 3)))
                val_metric_epochs.update({k: val_metric_value})
                print('====================================================')
                if self.use_attention:
                    saver.save(sess, r"C:/Users/sinad/Desktop/CATT_Intern/tenserflow_results/CNN-Attention", global_step=k)
                else:
                    saver.save(sess, r"C:/Users/sinad/Desktop/CATT_Intern/tenserflow_results/CNN-no-Attention",
                               global_step=k)
                if all([val_metric_epochs[k] - val_metric_epochs[k - 1] < self.val_threshold, val_metric_epochs[k] - val_metric_epochs[k - 2] < self.val_threshold]):
                    break
                else:
                    k += 1

            print("Val {} Over Epochs: {}".format(self.val_metric, val_metric_epochs))
            max_metric = max(val_metric_epochs.items(), key=lambda k: k[1])
            if self.use_attention:
                saver.restore(sess, r"C:/Users/sinad/Desktop/CATT_Intern/tenserflow_results/CNN-Attention-"+str(max_metric[0]))
            else:
                saver.save(sess, r"C:/Users/sinad/Desktop/CATT_Intern/tenserflow_results/CNN-no-Attention", global_step=k)

            y_pred, pred_prob = self.prediction_prob(Test_X, classifier_output, input_layer, sess)

            return y_pred, pred_prob, Test_Y_ori
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

    def training_all_folds(self, kfold_dataset):
        test_accuracy_fold = []
        ave_roc_auc_fold = []
        ave_recall_fold = []
        std_recall_fold = []

        confusion_matrix_fold = []
        cls_report = []
        for fold in kfold_dataset:
            y_pred, pred_prob, Test_Y_ori = self.training(fold)
            Test_Y = keras.utils.to_categorical(Test_Y_ori, num_classes=len(np.unique(Test_Y_ori)))  # one-hot vector of y
            test_accuracy_fold.append(accuracy_score(Test_Y_ori, y_pred))
            #ave_roc_auc_fold.append(roc_auc_score(Test_Y, pred_prob, average='macro'))
            ave_roc_auc_fold.append(DeepLearningModel.categorical_roc_score(Test_Y_ori, pred_prob))

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

        print('CNN Accuracy:  Mean {}, std {}'.format(mean_acc, std_acc))
        print('CNN AUC Score:: Mean {}, std {}'.format(mean_auc, std_auc))
        print('CNN Average Recall Score:: Mean {}, std {}'.format(mean_ave_recall, std_ave_recall))
        print('CNN STD Recall Score:: Mean {}, std {}'.format(mean_std_recall, std_std_recall))

        for i in range(len(confusion_matrix_fold)):
            print('Confusion Matrix for fold {} : {}'.format(i, confusion_matrix_fold[i]))
            print('\n')

        print('\n')


if __name__ == '__main__':
    # initializer: tf.contrib.layers.variance_scaling_initializer(), tf.contrib.layers.xavier_initializer(uniform=True, seed=7)
    deep_learning_model = DeepLearningModel(num_filter=[100], num_dense=0, num_conv_set=1, use_attention=1, attention_MLP=0, initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=7),
                                        kernel_size=[2], strides=1, activation=tf.nn.relu, learning_rate=0.001, padding='valid', epochs=20, batch_size=128, num_fold_training='All', val_threshold=0.0001,
                                            make_balance_dataset=0, make_balance_batch=1, alpha=0.8, weighted_logits=[1., 1.], val_metric='recall')
    filename = 'kfold_dataset_DL_light&heavy_mean_std_OSRM_2class.pickle'
    with open(filename, 'rb') as f:
        kfold_dataset = pickle.load(f)

    def extract_feature(kfold_dataset):
        for i in range(len(kfold_dataset)):
            kfold_dataset[i][0] = kfold_dataset[i][0][:, :, :7, :]
            kfold_dataset[i][2] = kfold_dataset[i][2][:, :, :7, :]
            kfold_dataset[i][4] = kfold_dataset[i][4][:, :, :7, :]
        return kfold_dataset

    #kfold_dataset = extract_feature(kfold_dataset)


    class_1 = len(np.where(kfold_dataset[0][5] == 0)[0])
    class_2 = len(np.where(kfold_dataset[0][5] == 1)[0])

    deep_learning_model.training_all_folds(kfold_dataset)

# Settings
a=1