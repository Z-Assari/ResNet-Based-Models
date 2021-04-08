import tensorflow as tf
import math
import numpy as np
from numpy import load
from random import sample
from scipy.io import loadmat, savemat
from sklearn import metrics
from sklearn.metrics import confusion_matrix

########################################################################################################################
#****************************************************** Functions ******************************************************
def splitting_cv(masses_num,k):
    # Data splitting for k-fold cross validation
    folds = math.floor(masses_num/k)*np.ones((k,1), dtype=int)
    n = masses_num-(k*math.floor(masses_num/k))
    i = 0
    while 0 < n:
        folds[i,0] = folds[i,0] + 1
        i = i + 1
        n = n - 1
    return folds


def test_train_masses_num(randsampled_masses, splitting_masses, test_stage):
    test_masses_num = np.array(randsampled_masses[np.sum(splitting_masses[0:test_stage-1,0]):np.sum(splitting_masses[0:test_stage,0])])
    train_masses_num = np.concatenate((np.array(randsampled_masses[0:np.sum(splitting_masses[0:test_stage-1,0])]), np.array(randsampled_masses[np.sum(splitting_masses[0:test_stage,0]):])))

    test_masses_num = test_masses_num.astype(int)
    train_masses_num = train_masses_num.astype(int)
    return test_masses_num, train_masses_num


def dataset(view_type, resize_parameter, benign_masses_num, malignant_masses_num):
    Address = r'C:\Dataset/' + str(resize_parameter) + '_' + str(resize_parameter) + '/'
    Dataset_Load = loadmat(Address + 'Dataset_' + view_type + '_' + str(resize_parameter) + '_' + str(resize_parameter) + '.mat')

    # Benign:
    Dataset_Matrix_Benign = Dataset_Load['Matrix_' + view_type + '_' + str(resize_parameter) + '_' + str(resize_parameter) + '_Benign']
    Dataset_Label_Vector_Benign = Dataset_Load['Label_Vector_' + view_type + '_' + str(resize_parameter) + '_' + str(resize_parameter) + '_Benign']

    Matrix_Benign = np.zeros((benign_masses_num.shape[0], Dataset_Matrix_Benign.shape[1]))
    Label_Vector_Benign = np.zeros((benign_masses_num.shape[0], 2))
    for i in range(benign_masses_num.shape[0]):
        Matrix_Benign[i, :] = Dataset_Matrix_Benign[benign_masses_num[i]-1, :]
        Label_Vector_Benign[i, :] = Dataset_Label_Vector_Benign[benign_masses_num[i]-1, :]

    # Malignant:
    Dataset_Matrix_Malignant = Dataset_Load['Matrix_' + view_type + '_' + str(resize_parameter) + '_' + str(resize_parameter) + '_Malignant']
    Dataset_Label_Vector_Malignant = Dataset_Load['Label_Vector_' + view_type + '_' + str(resize_parameter) + '_' + str(resize_parameter) + '_Malignant']

    Matrix_Malignant = np.zeros((malignant_masses_num.shape[0], Dataset_Matrix_Malignant.shape[1]))
    Label_Vector_Malignant = np.zeros((malignant_masses_num.shape[0], 2))
    for i in range(malignant_masses_num.shape[0]):
        Matrix_Malignant[i, :] = Dataset_Matrix_Malignant[malignant_masses_num[i]-1, :]
        Label_Vector_Malignant[i, :] = Dataset_Label_Vector_Malignant[malignant_masses_num[i]-1, :]

    # Concatenation:
    matrix = np.zeros(((benign_masses_num.shape[0] + malignant_masses_num.shape[0]), Dataset_Matrix_Malignant.shape[1]))
    label_vector = np.zeros(((benign_masses_num.shape[0] + malignant_masses_num.shape[0]), 2))
    matrix = np.concatenate((Matrix_Benign, Matrix_Malignant), axis=0)
    matrix = matrix.astype(np.float32) / 255
    label_vector = np.concatenate((Label_Vector_Benign, Label_Vector_Malignant), axis=0)
    return matrix, label_vector


def dataset_augmentation_shuffling(view_type, resize_parameter, augmentation_parameter, benign_masses_num, malignant_masses_num, Random_Sampling):
    Address = r'C:\Dataset/' + str(resize_parameter) + '_' + str(resize_parameter) + '/'
    Dataset_Load = loadmat(Address + 'Augmented_Dataset_' + view_type + '_' + str(resize_parameter) + '_' + str(resize_parameter) + '.mat')

    # Benign:
    Dataset_Matrix_Benign = Dataset_Load['Augmented_Matrix_' + view_type + '_' + str(resize_parameter) + '_' + str(resize_parameter) + '_Benign']
    Dataset_Label_Vector_Benign = Dataset_Load['Augmented_Label_Vector_' + view_type + '_' + str(resize_parameter) + '_' + str(resize_parameter) + '_Benign']

    Matrix_Benign = np.zeros(((benign_masses_num.shape[0] * augmentation_parameter), Dataset_Matrix_Benign.shape[1]))
    Label_Vector_Benign = np.zeros(((benign_masses_num.shape[0] * augmentation_parameter), 2))
    m = 0
    for i in range(benign_masses_num.shape[0]):
        for j in range(augmentation_parameter):
            Num = (augmentation_parameter * (benign_masses_num[i]-1)) + j
            Matrix_Benign[m, :] = Dataset_Matrix_Benign[Num, :]
            Label_Vector_Benign[m, :] = Dataset_Label_Vector_Benign[Num, :]
            m = m + 1

    # Malignant:
    Dataset_Matrix_Malignant = Dataset_Load['Augmented_Matrix_' + view_type + '_' + str(resize_parameter) + '_' + str(resize_parameter) + '_Malignant']
    Dataset_Label_Vector_Malignant = Dataset_Load['Augmented_Label_Vector_' + view_type + '_' + str(resize_parameter) + '_' + str(resize_parameter) + '_Malignant']

    Matrix_Malignant = np.zeros(((malignant_masses_num.shape[0] * augmentation_parameter), Dataset_Matrix_Malignant.shape[1]))
    Label_Vector_Malignant = np.zeros(((malignant_masses_num.shape[0] * augmentation_parameter), 2))
    m = 0
    for i in range(malignant_masses_num.shape[0]):
        for j in range(augmentation_parameter):
            Num = (augmentation_parameter * (malignant_masses_num[i]-1)) + j
            Matrix_Malignant[m, :] = Dataset_Matrix_Malignant[Num, :]
            Label_Vector_Malignant[m, :] = Dataset_Label_Vector_Malignant[Num, :]
            m = m + 1

    # Concatenation:
    matrix = np.zeros((((benign_masses_num.shape[0] + malignant_masses_num.shape[0]) * augmentation_parameter), Dataset_Matrix_Malignant.shape[1]))
    label_vector = np.zeros((((benign_masses_num.shape[0] + malignant_masses_num.shape[0]) * augmentation_parameter), 2))
    matrix = np.concatenate((Matrix_Benign, Matrix_Malignant), axis=0)
    label_vector = np.concatenate((Label_Vector_Benign, Label_Vector_Malignant), axis=0)

    # Shuffling:
    augmented_matrix = np.zeros((((benign_masses_num.shape[0] + malignant_masses_num.shape[0]) * augmentation_parameter), Dataset_Matrix_Malignant.shape[1]))
    augmented_label_vector = np.zeros((((benign_masses_num.shape[0] + malignant_masses_num.shape[0]) * augmentation_parameter), 2))
    for k in range(matrix.shape[0]):
        augmented_matrix[k, :] = matrix[Random_Sampling[k]-1, :]
        augmented_label_vector[k, :] = label_vector[Random_Sampling[k] - 1, :]
    return augmented_matrix, augmented_label_vector


def conv_layer_ago(input_volume, filters, channels, strides):
    w_conv = tf.Variable(tf.truncated_normal([filters, filters, input_volume.shape[3].value, channels], stddev=0.1))
    b_conv = tf.Variable(tf.constant(0.1, shape=[channels]))
    conv = tf.nn.conv2d(input_volume, w_conv, strides=[1, strides, strides, 1], padding='SAME') + b_conv
    return conv


def conv_layer(input_volume, filters, channels, strides, conv_index, bias_term):
    # "ResNet50.npy":
    ResNet50 = {}
    ResNet50 = load(open("ResNet50.npy", "rb"), encoding="latin1").item()
    if bias_term is 'True':
        Conv_W = ResNet50[conv_index + '/W']
        Conv_W = Conv_W[:, :, 1:2, :]
        Conv_B = ResNet50[conv_index + '/b']
        w_conv = tf.Variable(Conv_W.astype(np.float32))
        b_conv = tf.Variable(Conv_B.astype(np.float32))
        conv = tf.nn.conv2d(input_volume, w_conv, strides=[1, strides, strides, 1], padding='SAME') + b_conv
        return conv
    else:
        Conv_W = ResNet50[conv_index + '/W']
        w_conv = tf.Variable(Conv_W.astype(np.float32))
        conv = tf.nn.conv2d(input_volume, w_conv, strides=[1, strides, strides, 1], padding='SAME')
        return conv


def fc_layer(input_volume, output_neurons_num):
    shape_input_volume = input_volume.shape
    if len(shape_input_volume) > 2:
        neurons_reshaped_num_fc = input_volume.shape[1].value * input_volume.shape[2].value * input_volume.shape[3].value
        reshaped_input_volume = tf.reshape(input_volume, [-1, neurons_reshaped_num_fc])
        w_fc = tf.Variable(tf.truncated_normal([neurons_reshaped_num_fc, output_neurons_num], stddev=0.1))
        b_fc = tf.Variable(tf.constant(0.1, shape=[output_neurons_num]))
        fc = tf.matmul(reshaped_input_volume, w_fc) + b_fc
        return fc
    else:
        w_fc = tf.Variable(tf.truncated_normal([input_volume.shape[1].value, output_neurons_num], stddev=0.1))
        b_fc = tf.Variable(tf.constant(0.1, shape=[output_neurons_num]))
        fc = tf.matmul(input_volume, w_fc) + b_fc
        return fc


def bn_layer(input_volume, update_ops_collection, bn_index, is_training):
    decay = 0.9997
    epsilon = 0.001
    ResNet50 = {}
    ResNet50 = load(open("ResNet50.npy", "rb"), encoding="latin1").item()

    int_beta = ResNet50[bn_index + '/beta']
    int_gamma = ResNet50[bn_index + '/gamma']
    int_mean = ResNet50[bn_index + '/mean/EMA']
    int_variance = ResNet50[bn_index + '/variance/EMA']

    beta = tf.Variable(int_beta.astype(np.float32))
    gamma = tf.Variable(int_gamma.astype(np.float32))
    pop_mean = tf.Variable(int_mean.astype(np.float32))
    pop_variance = tf.Variable(int_variance.astype(np.float32))

    if is_training is True:
        bn = tf.nn.batch_normalization(input_volume, pop_mean, pop_variance, beta, gamma, epsilon)
        return bn
    else:
        input_volume_shape = input_volume.get_shape()
        axis = list(range(len(input_volume_shape) - 1))
        batch_mean, batch_variance = tf.nn.moments(input_volume, axis)
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_variance = tf.assign(pop_variance, pop_variance * decay + batch_variance * (1 - decay))
        with tf.control_dependencies([train_mean, train_variance]):
            bn = tf.nn.batch_normalization(input_volume, batch_mean, batch_variance, beta, gamma, epsilon)
            return bn


def next_batch(Minibatch_Size, Epoch_Num, Matrix_View1, Matrix_View2, Matrix_View3, Label_Vector):
    num_examples = Matrix_View1.shape[0]
    End_ago = ((Epoch_Num - 1) * Minibatch_Size) % num_examples
    Matrix_View1_Minibatch = np.zeros((Minibatch_Size, Matrix_View1.shape[1]))
    Matrix_View2_Minibatch = np.zeros((Minibatch_Size, Matrix_View2.shape[1]))
    Matrix_View3_Minibatch = np.zeros((Minibatch_Size, Matrix_View3.shape[1]))
    Label_Vector_Minibatch = np.zeros((Minibatch_Size, 2))

    if ((End_ago + Minibatch_Size) < num_examples + 1):
        Matrix_View1_Minibatch = Matrix_View1[End_ago:End_ago + Minibatch_Size, :]
        Matrix_View1_Minibatch = Matrix_View1_Minibatch.astype(np.float32) / 255
        Matrix_View2_Minibatch = Matrix_View2[End_ago:End_ago + Minibatch_Size, :]
        Matrix_View2_Minibatch = Matrix_View2_Minibatch.astype(np.float32) / 255
        Matrix_View3_Minibatch = Matrix_View3[End_ago:End_ago + Minibatch_Size, :]
        Matrix_View3_Minibatch = Matrix_View3_Minibatch.astype(np.float32) / 255

        Label_Vector_Minibatch = Label_Vector[End_ago:End_ago + Minibatch_Size, :]
        return Matrix_View1_Minibatch, Matrix_View2_Minibatch, Matrix_View3_Minibatch, Label_Vector_Minibatch
    else:
        Matrix1_View1 = Matrix_View1[End_ago:, :]
        Matrix2_View1 = Matrix_View1[0:Minibatch_Size - (num_examples - End_ago), :]
        Matrix_View1_Minibatch[:num_examples - End_ago, :] = Matrix1_View1
        Matrix_View1_Minibatch[num_examples - End_ago:, :] = Matrix2_View1
        Matrix_View1_Minibatch = Matrix_View1_Minibatch.astype(np.float32) / 255

        Matrix1_View2 = Matrix_View2[End_ago:, :]
        Matrix2_View2 = Matrix_View2[0:Minibatch_Size - (num_examples - End_ago), :]
        Matrix_View2_Minibatch[:num_examples - End_ago, :] = Matrix1_View2
        Matrix_View2_Minibatch[num_examples - End_ago:, :] = Matrix2_View2
        Matrix_View2_Minibatch = Matrix_View2_Minibatch.astype(np.float32) / 255

        Matrix1_View3 = Matrix_View3[End_ago:, :]
        Matrix2_View3 = Matrix_View3[0:Minibatch_Size - (num_examples - End_ago), :]
        Matrix_View3_Minibatch[:num_examples - End_ago, :] = Matrix1_View3
        Matrix_View3_Minibatch[num_examples - End_ago:, :] = Matrix2_View3
        Matrix_View3_Minibatch = Matrix_View3_Minibatch.astype(np.float32) / 255

        Vector1 = Label_Vector[End_ago:, :]
        Vector2 = Label_Vector[0:Minibatch_Size - (num_examples - End_ago), :]
        Label_Vector_Minibatch[:num_examples - End_ago, :] = Vector1
        Label_Vector_Minibatch[num_examples - End_ago:, :] = Vector2

        return Matrix_View1_Minibatch, Matrix_View2_Minibatch, Matrix_View3_Minibatch, Label_Vector_Minibatch


def save_all(View_Name, k, Address):
    # Model_US_MG_CC_MG_MLO.mat
    Result = {}
    for test_stage in range(1, k+1):
        a = {}
        a = loadmat(Address + '\Stage_' + str(test_stage) + '_y_Gold_y_Deep_NN_' + View_Name[0] + '_' + View_Name[1] + '_' + View_Name[2] + '.mat')
        Result['Stage_' + str(test_stage) + '_y_Gold'] = a['Stage_' + str(test_stage) + '_y_Gold']
        Result['Stage_' + str(test_stage) + '_y_Deep_NN'] = a['Stage_' + str(test_stage) + '_y_Deep_NN']

        Result['Stage_' + str(test_stage) + '_Test_Benign_Masses_Num'] = a['Stage_' + str(test_stage) + '_Test_Benign_Masses_Num']
        Result['Stage_' + str(test_stage) + '_Train_Benign_Masses_Num'] = a['Stage_' + str(test_stage) + '_Train_Benign_Masses_Num']

        Result['Stage_' + str(test_stage) + '_Test_Malignant_Masses_Num'] = a['Stage_' + str(test_stage) + '_Test_Malignant_Masses_Num']
        Result['Stage_' + str(test_stage) + '_Train_Malignant_Masses_Num'] = a['Stage_' + str(test_stage) + '_Train_Malignant_Masses_Num']

        Result['randsampled_benign_masses'] = a['randsampled_benign_masses']
        Result['randsampled_malignant_masses'] = a['randsampled_malignant_masses']

        Result['splitting_benign_masses'] = a['splitting_benign_masses']
        Result['splitting_malignant_masses'] = a['splitting_malignant_masses']
    Name = Address + '\Model_' + View_Name[0] + '_' + View_Name[1] + '_' + View_Name[2] + '.mat'
    savemat(Name, Result)
    return


def evaluation(View_Name, k, Address):
    D = loadmat(Address + '\Model_' + View_Name[0] + '_' + View_Name[1] + '_' + View_Name[2] + '.mat')
    SE = np.zeros((k,))
    SP = np.zeros((k,))
    F1 = np.zeros((k,))
    AUC = np.zeros((k,))
    AC = np.zeros((k,))
    m = 0
    for test_stage in range(1, k+1):
        y_Gold = D['Stage_' + str(test_stage) + '_y_Gold']
        y_Deep_NN = D['Stage_' + str(test_stage) + '_y_Deep_NN']
        Y_true = tf.argmax(y_Gold, 1)
        Y_pred = tf.argmax(y_Deep_NN, 1)
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
            result_Y_true = session.run(Y_true)
            result_Y_pred = session.run(Y_pred)
            tn, fp, fn, tp = confusion_matrix(result_Y_true, result_Y_pred).ravel()
            AC[m] = metrics.accuracy_score(result_Y_true, result_Y_pred)
            SE[m] = metrics.recall_score(result_Y_true, result_Y_pred)
            F1[m] = metrics.f1_score(result_Y_true, result_Y_pred)
            SP[m] = tn / float(fp + tn)
            y_pred_prob = y_Deep_NN[:, 1]
            AUC[m] = metrics.roc_auc_score(result_Y_true, y_pred_prob)
            m = m + 1
    a = {}
    Evaluation_Metrics = np.zeros((1, 5))
    Evaluation_Metrics[0, :] = np.array([round(np.mean(SE), 3), round(np.mean(SP), 3), round(np.mean(F1), 3), round(np.mean(AUC), 3), round(np.mean(AC), 3)])
    a['Evaluation_Metrics'] = Evaluation_Metrics

    a['AC'] = AC
    a['AUC'] = AUC
    a['SE'] = SE
    a['SP'] = SP
    a['F1'] = F1

    Name = Address + '\Evaluation_Metrics_' + View_Name[0] + '_' + View_Name[1] + '_' + View_Name[2] + '.mat'
    savemat(Name, a)
    return


def part1(input_volume, update_ops_collection, is_training):
    Conv1 = tf.nn.relu(bn_layer(conv_layer(input_volume, 7, 64, 2, 'conv1', 'True'), update_ops_collection, 'bn_conv1', is_training))
    MaxPool = tf.nn.max_pool(Conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    return MaxPool


def part2(input_volume, update_ops_collection, is_training):
    # res2: 3 × Bottleneck building block1:
    Conv_res2a_branch2a = tf.nn.relu(bn_layer(conv_layer(input_volume, 1, 64, 1, 'res2a_branch2a', 'False'), update_ops_collection, 'bn2a_branch2a', is_training))
    Conv_res2a_branch2b = tf.nn.relu(bn_layer(conv_layer(Conv_res2a_branch2a, 3, 64, 1, 'res2a_branch2b', 'False'), update_ops_collection, 'bn2a_branch2b', is_training))
    Conv_res2a_branch2c = bn_layer(conv_layer(Conv_res2a_branch2b, 1, 256, 1, 'res2a_branch2c', 'False'), update_ops_collection, 'bn2a_branch2c', is_training)
    Conv_res2a_branch1 = bn_layer(conv_layer(input_volume, 1, 256, 1, 'res2a_branch1', 'False'), update_ops_collection, 'bn2a_branch1', is_training)
    Addition1_res2 = tf.nn.relu(Conv_res2a_branch2c + Conv_res2a_branch1)

    Conv_res2b_branch2a = tf.nn.relu(bn_layer(conv_layer(Addition1_res2, 1, 64, 1, 'res2b_branch2a', 'False'), update_ops_collection, 'bn2b_branch2a', is_training))
    Conv_res2b_branch2b = tf.nn.relu(bn_layer(conv_layer(Conv_res2b_branch2a, 3, 64, 1, 'res2b_branch2b', 'False'), update_ops_collection, 'bn2b_branch2b', is_training))
    Conv_res2b_branch2c = bn_layer(conv_layer(Conv_res2b_branch2b, 1, 256, 1, 'res2b_branch2c', 'False'), update_ops_collection, 'bn2b_branch2c', is_training)
    Addition2_res2 = tf.nn.relu(Conv_res2b_branch2c + Addition1_res2)

    Conv_res2c_branch2a = tf.nn.relu(bn_layer(conv_layer(Addition2_res2, 1, 64, 1, 'res2c_branch2a', 'False'), update_ops_collection, 'bn2c_branch2a', is_training))
    Conv_res2c_branch2b = tf.nn.relu(bn_layer(conv_layer(Conv_res2c_branch2a, 3, 64, 1, 'res2c_branch2b', 'False'), update_ops_collection, 'bn2c_branch2b', is_training))
    Conv_res2c_branch2c = bn_layer(conv_layer(Conv_res2c_branch2b, 1, 256, 1, 'res2c_branch2c', 'False'), update_ops_collection, 'bn2c_branch2c', is_training)
    Addition3_res2 = tf.nn.relu(Conv_res2c_branch2c + Addition2_res2)
    return Addition3_res2


def part3(input_volume, update_ops_collection, is_training):
    # res3: 4 × Bottleneck building block2:
    Conv_res3a_branch2a = tf.nn.relu(bn_layer(conv_layer(input_volume, 1, 128, 2, 'res3a_branch2a', 'False'), update_ops_collection, 'bn3a_branch2a', is_training))
    Conv_res3a_branch2b = tf.nn.relu(bn_layer(conv_layer(Conv_res3a_branch2a, 3, 128, 1, 'res3a_branch2b', 'False'), update_ops_collection, 'bn3a_branch2b', is_training))
    Conv_res3a_branch2c = bn_layer(conv_layer(Conv_res3a_branch2b, 1, 512, 1, 'res3a_branch2c', 'False'), update_ops_collection, 'bn3a_branch2c', is_training)
    Conv_res3a_branch1 = bn_layer(conv_layer(input_volume, 1, 512, 2, 'res3a_branch1', 'False'), update_ops_collection, 'bn3a_branch1', is_training)
    Addition1_res3 = tf.nn.relu(Conv_res3a_branch2c + Conv_res3a_branch1)

    Conv_res3b_branch2a = tf.nn.relu(bn_layer(conv_layer(Addition1_res3, 1, 128, 1, 'res3b_branch2a', 'False'), update_ops_collection, 'bn3b_branch2a', is_training))
    Conv_res3b_branch2b = tf.nn.relu(bn_layer(conv_layer(Conv_res3b_branch2a, 3, 128, 1, 'res3b_branch2b', 'False'), update_ops_collection, 'bn3b_branch2b', is_training))
    Conv_res3b_branch2c = bn_layer(conv_layer(Conv_res3b_branch2b, 1, 512, 1, 'res3b_branch2c', 'False'), update_ops_collection, 'bn3b_branch2c', is_training)
    Addition2_res3 = tf.nn.relu(Conv_res3b_branch2c + Addition1_res3)

    Conv_res3c_branch2a = tf.nn.relu(bn_layer(conv_layer(Addition2_res3, 1, 128, 1, 'res3c_branch2a', 'False'), update_ops_collection, 'bn3c_branch2a', is_training))
    Conv_res3c_branch2b = tf.nn.relu(bn_layer(conv_layer(Conv_res3c_branch2a, 3, 128, 1, 'res3c_branch2b', 'False'), update_ops_collection, 'bn3c_branch2b', is_training))
    Conv_res3c_branch2c = bn_layer(conv_layer(Conv_res3c_branch2b, 1, 512, 1, 'res3c_branch2c', 'False'), update_ops_collection, 'bn3c_branch2c', is_training)
    Addition3_res3 = tf.nn.relu(Conv_res3c_branch2c + Addition2_res3)

    Conv_res3d_branch2a = tf.nn.relu(bn_layer(conv_layer(Addition3_res3, 1, 128, 1, 'res3d_branch2a', 'False'), update_ops_collection, 'bn3d_branch2a', is_training))
    Conv_res3d_branch2b = tf.nn.relu(bn_layer(conv_layer(Conv_res3d_branch2a, 3, 128, 1, 'res3d_branch2b', 'False'), update_ops_collection, 'bn3d_branch2b', is_training))
    Conv_res3d_branch2c = bn_layer(conv_layer(Conv_res3d_branch2b, 1, 512, 1, 'res3d_branch2c', 'False'), update_ops_collection, 'bn3d_branch2c', is_training)
    Addition4_res3 = tf.nn.relu(Conv_res3d_branch2c + Addition3_res3)
    return Addition4_res3


def part4(input_volume, update_ops_collection, is_training):
    # res4: 6 × Bottleneck building block3:
    Conv_res4a_branch2a = tf.nn.relu(bn_layer(conv_layer(input_volume, 1, 256, 2, 'res4a_branch2a', 'False'), update_ops_collection, 'bn4a_branch2a', is_training))
    Conv_res4a_branch2b = tf.nn.relu(bn_layer(conv_layer(Conv_res4a_branch2a, 3, 256, 1, 'res4a_branch2b', 'False'), update_ops_collection, 'bn4a_branch2b', is_training))
    Conv_res4a_branch2c = bn_layer(conv_layer(Conv_res4a_branch2b, 1, 1024, 1, 'res4a_branch2c', 'False'), update_ops_collection, 'bn4a_branch2c', is_training)
    Conv_res4a_branch1 = bn_layer(conv_layer(input_volume, 1, 1024, 2, 'res4a_branch1', 'False'), update_ops_collection, 'bn4a_branch1', is_training)
    Addition1_res4 = tf.nn.relu(Conv_res4a_branch2c + Conv_res4a_branch1)

    Conv_res4b_branch2a = tf.nn.relu(bn_layer(conv_layer(Addition1_res4, 1, 256, 1, 'res4b_branch2a', 'False'), update_ops_collection, 'bn4b_branch2a', is_training))
    Conv_res4b_branch2b = tf.nn.relu(bn_layer(conv_layer(Conv_res4b_branch2a, 3, 256, 1, 'res4b_branch2b', 'False'), update_ops_collection, 'bn4b_branch2b', is_training))
    Conv_res4b_branch2c = bn_layer(conv_layer(Conv_res4b_branch2b, 1, 1024, 1, 'res4b_branch2c', 'False'), update_ops_collection, 'bn4b_branch2c', is_training)
    Addition2_res4 = tf.nn.relu(Conv_res4b_branch2c + Addition1_res4)

    Conv_res4c_branch2a = tf.nn.relu(bn_layer(conv_layer(Addition2_res4, 1, 256, 1, 'res4c_branch2a', 'False'), update_ops_collection, 'bn4c_branch2a', is_training))
    Conv_res4c_branch2b = tf.nn.relu(bn_layer(conv_layer(Conv_res4c_branch2a, 3, 256, 1, 'res4c_branch2b', 'False'), update_ops_collection, 'bn4c_branch2b', is_training))
    Conv_res4c_branch2c = bn_layer(conv_layer(Conv_res4c_branch2b, 1, 1024, 1, 'res4c_branch2c', 'False'), update_ops_collection, 'bn4c_branch2c', is_training)
    Addition3_res4 = tf.nn.relu(Conv_res4c_branch2c + Addition2_res4)

    Conv_res4d_branch2a = tf.nn.relu(bn_layer(conv_layer(Addition3_res4, 1, 256, 1, 'res4d_branch2a', 'False'), update_ops_collection, 'bn4d_branch2a', is_training))
    Conv_res4d_branch2b = tf.nn.relu(bn_layer(conv_layer(Conv_res4d_branch2a, 3, 256, 1, 'res4d_branch2b', 'False'), update_ops_collection, 'bn4d_branch2b', is_training))
    Conv_res4d_branch2c = bn_layer(conv_layer(Conv_res4d_branch2b, 1, 1024, 1, 'res4d_branch2c', 'False'), update_ops_collection, 'bn4d_branch2c', is_training)
    Addition4_res4 = tf.nn.relu(Conv_res4d_branch2c + Addition3_res4)

    Conv_res4e_branch2a = tf.nn.relu(bn_layer(conv_layer(Addition4_res4, 1, 256, 1, 'res4e_branch2a', 'False'), update_ops_collection, 'bn4e_branch2a', is_training))
    Conv_res4e_branch2b = tf.nn.relu(bn_layer(conv_layer(Conv_res4e_branch2a, 3, 256, 1, 'res4e_branch2b', 'False'), update_ops_collection, 'bn4e_branch2b', is_training))
    Conv_res4e_branch2c = bn_layer(conv_layer(Conv_res4e_branch2b, 1, 1024, 1, 'res4e_branch2c', 'False'), update_ops_collection, 'bn4e_branch2c', is_training)
    Addition5_res4 = tf.nn.relu(Conv_res4e_branch2c + Addition4_res4)

    Conv_res4f_branch2a = tf.nn.relu(bn_layer(conv_layer(Addition5_res4, 1, 256, 1, 'res4f_branch2a', 'False'), update_ops_collection, 'bn4f_branch2a', is_training))
    Conv_res4f_branch2b = tf.nn.relu(bn_layer(conv_layer(Conv_res4f_branch2a, 3, 256, 1, 'res4f_branch2b', 'False'), update_ops_collection, 'bn4f_branch2b', is_training))
    Conv_res4f_branch2c = bn_layer(conv_layer(Conv_res4f_branch2b, 1, 1024, 1, 'res4f_branch2c', 'False'), update_ops_collection, 'bn4f_branch2c', is_training)
    Addition6_res4 = tf.nn.relu(Conv_res4f_branch2c + Addition5_res4)
    return Addition6_res4


def part5(input_volume, update_ops_collection, is_training):
    # res5: 3 × Bottleneck building block4:
    Conv_res5a_branch2a = tf.nn.relu(bn_layer(conv_layer(input_volume, 1, 512, 2, 'res5a_branch2a', 'False'), update_ops_collection, 'bn5a_branch2a', is_training))
    Conv_res5a_branch2b = tf.nn.relu(bn_layer(conv_layer(Conv_res5a_branch2a, 3, 512, 1, 'res5a_branch2b', 'False'), update_ops_collection, 'bn5a_branch2b', is_training))
    Conv_res5a_branch2c = bn_layer(conv_layer(Conv_res5a_branch2b, 1, 2048, 1, 'res5a_branch2c', 'False'), update_ops_collection, 'bn5a_branch2c', is_training)
    Conv_res5a_branch1 = bn_layer(conv_layer(input_volume, 1, 2048, 2, 'res5a_branch1', 'False'), update_ops_collection, 'bn5a_branch1', is_training)
    Addition1_res5 = tf.nn.relu(Conv_res5a_branch2c + Conv_res5a_branch1)

    Conv_res5b_branch2a = tf.nn.relu(bn_layer(conv_layer(Addition1_res5, 1, 512, 1, 'res5b_branch2a', 'False'), update_ops_collection, 'bn5b_branch2a', is_training))
    Conv_res5b_branch2b = tf.nn.relu(bn_layer(conv_layer(Conv_res5b_branch2a, 3, 512, 1, 'res5b_branch2b', 'False'), update_ops_collection, 'bn5b_branch2b', is_training))
    Conv_res5b_branch2c = bn_layer(conv_layer(Conv_res5b_branch2b, 1, 2048, 1, 'res5b_branch2c', 'False'), update_ops_collection, 'bn5b_branch2c', is_training)
    Addition2_res5 = tf.nn.relu(Conv_res5b_branch2c + Addition1_res5)

    Conv_res5c_branch2a = tf.nn.relu(bn_layer(conv_layer(Addition2_res5, 1, 512, 1, 'res5c_branch2a', 'False'), update_ops_collection, 'bn5c_branch2a', is_training))
    Conv_res5c_branch2b = tf.nn.relu(bn_layer(conv_layer(Conv_res5c_branch2a, 3, 512, 1, 'res5c_branch2b', 'False'), update_ops_collection, 'bn5c_branch2b', is_training))
    Conv_res5c_branch2c = bn_layer(conv_layer(Conv_res5c_branch2b, 1, 2048, 1, 'res5c_branch2c', 'False'), update_ops_collection, 'bn5c_branch2c', is_training)
    Addition3_res5 = tf.nn.relu(Conv_res5c_branch2c + Addition2_res5)

    AvgPool = tf.reduce_mean(Addition3_res5, axis=[1, 2])
    Before_Softmax = fc_layer(AvgPool, 2)
    return Before_Softmax


def model(X_Image1, X_Image2, X_Image3, update_ops_collection, is_training):
    # ***** ResNet50_4_5 ***** : Branch1x: [Start-Part4), Branch2y: [Part4-Part5), Branch3: [Part5-End]
    # ******************************************************************************************************************
    # Branch11: [Start-Part4) => (Input: X_Image1, Output: Addition4_res3_Branch11)
    MaxPool_Branch11 = part1(X_Image1, update_ops_collection, is_training)
    Addition3_res2_Branch11 = part2(MaxPool_Branch11, update_ops_collection, is_training)
    Addition4_res3_Branch11 = part3(Addition3_res2_Branch11, update_ops_collection, is_training)
    # Branch12: [Start-Part4) => (Input: X_Image2, Output: Addition4_res3_Branch12)
    MaxPool_Branch12 = part1(X_Image2, update_ops_collection, is_training)
    Addition3_res2_Branch12 = part2(MaxPool_Branch12, update_ops_collection, is_training)
    Addition4_res3_Branch12 = part3(Addition3_res2_Branch12, update_ops_collection, is_training)
    # Branch13: [Start-Part4) => (Input: X_Image1, Output: Addition4_res3_Branch13)
    MaxPool_Branch13 = part1(X_Image1, update_ops_collection, is_training)
    Addition3_res2_Branch13 = part2(MaxPool_Branch13, update_ops_collection, is_training)
    Addition4_res3_Branch13 = part3(Addition3_res2_Branch13, update_ops_collection, is_training)
    # Branch14: [Start-Part4) => (Input: X_Image3, Output: Addition4_res3_Branch14)
    MaxPool_Branch14 = part1(X_Image3, update_ops_collection, is_training)
    Addition3_res2_Branch14 = part2(MaxPool_Branch14, update_ops_collection, is_training)
    Addition4_res3_Branch14 = part3(Addition3_res2_Branch14, update_ops_collection, is_training)
    # Branch15: [Start-Part4) => (Input: X_Image2, Output: Addition4_res3_Branch15)
    MaxPool_Branch15 = part1(X_Image2, update_ops_collection, is_training)
    Addition3_res2_Branch15 = part2(MaxPool_Branch15, update_ops_collection, is_training)
    Addition4_res3_Branch15 = part3(Addition3_res2_Branch15, update_ops_collection, is_training)
    # Branch16: [Start-Part4) => (Input: X_Image3, Output: Addition4_res3_Branch16)
    MaxPool_Branch16 = part1(X_Image3, update_ops_collection, is_training)
    Addition3_res2_Branch16 = part2(MaxPool_Branch16, update_ops_collection, is_training)
    Addition4_res3_Branch16 = part3(Addition3_res2_Branch16, update_ops_collection, is_training)
    # ******************************************************************************************************************
    # Concatenation21 (Addition4_res3_Branch11, Addition4_res3_Branch12) => Conv_ReLU_New21
    Concatenation21 = tf.concat([Addition4_res3_Branch11, Addition4_res3_Branch12], 3)
    Conv_ReLU_New21 = tf.nn.relu(conv_layer_ago(Concatenation21, 1, 512, 1))  # (1×1, 512, S=1)
    # Concatenation22 (Addition4_res3_Branch13, Addition4_res3_Branch14) => Conv_ReLU_New22
    Concatenation22 = tf.concat([Addition4_res3_Branch13, Addition4_res3_Branch14], 3)
    Conv_ReLU_New22 = tf.nn.relu(conv_layer_ago(Concatenation22, 1, 512, 1))  # (1×1, 512, S=1)
    # Concatenation23 (Addition4_res3_Branch15, Addition4_res3_Branch16) => Conv_ReLU_New23
    Concatenation23 = tf.concat([Addition4_res3_Branch15, Addition4_res3_Branch16], 3)
    Conv_ReLU_New23 = tf.nn.relu(conv_layer_ago(Concatenation23, 1, 512, 1))  # (1×1, 512, S=1)
    # ******************************************************************************************************************
    # Branch21: [Part4-Part5) => (Input: Conv_ReLU_New21, Output: Addition6_res4_Branch21)
    Addition6_res4_Branch21 = part4(Conv_ReLU_New21, update_ops_collection, is_training)
    # Branch22: [Part4-Part5) => (Input: Conv_ReLU_New22, Output: Addition6_res4_Branch22)
    Addition6_res4_Branch22 = part4(Conv_ReLU_New22, update_ops_collection, is_training)
    # Branch23: [Part4-Part5) => (Input: Conv_ReLU_New23, Output: Addition6_res4_Branch23)
    Addition6_res4_Branch23 = part4(Conv_ReLU_New23, update_ops_collection, is_training)
    # ******************************************************************************************************************
    # Concatenation (Addition6_res4_Branch21, Addition6_res4_Branch22, Addition6_res4_Branch23) => Conv_ReLU_New
    Concatenation = tf.concat([Addition6_res4_Branch21, Addition6_res4_Branch22, Addition6_res4_Branch23], 3)
    Conv_ReLU_New = tf.nn.relu(conv_layer_ago(Concatenation, 1, 1024, 1))  # (1×1, 1024, S=1)
    # ******************************************************************************************************************
    # Branch3: [Part5-End] => (Input: Conv_ReLU_New, Output: Before_Softmax)
    Before_Softmax = part5(Conv_ReLU_New, update_ops_collection, is_training)
    return Before_Softmax


def run_deep_model(View_Name, randsampled_benign_masses, randsampled_malignant_masses, splitting_benign_masses, splitting_malignant_masses, test_stage, Address):
    # ***** Initialization *****
    update_ops_collection = 'resnet_update_ops'
    resize_parameter = 100
    augmentation_parameter = 24
    Training_Diff = 0.00000001
    numEpochs = 10000
    Minibatch_Size = 20
    Initial_Learning_Rate = 5e-4  # 5e-3
    Weight_Decay = 0.0005

    # ***** Interactive Session *****
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))

    X1 = tf.placeholder(tf.float32, shape=[None, (resize_parameter * resize_parameter)])
    X2 = tf.placeholder(tf.float32, shape=[None, (resize_parameter * resize_parameter)])
    X3 = tf.placeholder(tf.float32, shape=[None, (resize_parameter * resize_parameter)])

    X_Image1 = tf.reshape(X1, [-1, resize_parameter, resize_parameter, 1])
    X_Image2 = tf.reshape(X2, [-1, resize_parameter, resize_parameter, 1])
    X_Image3 = tf.reshape(X3, [-1, resize_parameter, resize_parameter, 1])

    y_Gold = tf.placeholder(tf.float32, shape=[None, 2])
    is_training = tf.placeholder(tf.bool)

    # ***** Model *****
    Before_Softmax = model(X_Image1, X_Image2, X_Image3, update_ops_collection, is_training)
    y_Deep_NN = tf.nn.softmax(Before_Softmax)

    # ***** Training & Validation *****
    # Training strategy: Stochastic gradient descent (SGD) optimization method (Minibatch Gradient Descent)
    DecayedLearningRate = tf.train.exponential_decay(learning_rate=Initial_Learning_Rate, global_step=1, decay_steps=Minibatch_Size, decay_rate=(1 - (100 * Weight_Decay)), staircase=True)

    # Stop criterion: (1) Minimizing the Cross-entropy loss function (2) Training is stopped when the Loss function does not improve after 10 epochs.
    Loss_fun = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_Gold, logits=Before_Softmax))
    training_OP = tf.train.GradientDescentOptimizer(DecayedLearningRate).minimize(Loss_fun)

    sess.run(tf.global_variables_initializer())

    Test_Benign_Masses_Num, Train_Benign_Masses_Num = test_train_masses_num(randsampled_benign_masses, splitting_benign_masses, test_stage)
    Test_Malignant_Masses_Num, Train_Malignant_Masses_Num = test_train_masses_num(randsampled_malignant_masses, splitting_malignant_masses, test_stage)

    Random_Sampling = sample(list(range(1, ((len(Train_Benign_Masses_Num) + len(Train_Malignant_Masses_Num)) * augmentation_parameter) + 1)), ((len(Train_Benign_Masses_Num) + len(Train_Malignant_Masses_Num)) * augmentation_parameter))

    # US:
    Train_Augmented_Matrix_View1, Train_Augmented_Label_Vector = dataset_augmentation_shuffling(View_Name[0], resize_parameter, augmentation_parameter, Train_Benign_Masses_Num, Train_Malignant_Masses_Num, Random_Sampling)
    Test_Matrix_View1, Test_Label_Vector = dataset(View_Name[0], resize_parameter, Test_Benign_Masses_Num, Test_Malignant_Masses_Num)
    # MG_CC:
    Train_Augmented_Matrix_View2, Train_Augmented_Label_Vector_View2 = dataset_augmentation_shuffling(View_Name[1], resize_parameter, augmentation_parameter, Train_Benign_Masses_Num, Train_Malignant_Masses_Num, Random_Sampling)
    Test_Matrix_View2, Test_Label_Vector_View2 = dataset(View_Name[1], resize_parameter, Test_Benign_Masses_Num, Test_Malignant_Masses_Num)
    # MG_MLO:
    Train_Augmented_Matrix_View3, Train_Augmented_Label_Vector_View3 = dataset_augmentation_shuffling(View_Name[2], resize_parameter, augmentation_parameter, Train_Benign_Masses_Num, Train_Malignant_Masses_Num, Random_Sampling)
    Test_Matrix_View3, Test_Label_Vector_View3 = dataset(View_Name[2], resize_parameter, Test_Benign_Masses_Num, Test_Malignant_Masses_Num)

    cost = 0
    diff = 1
    for Epoch_Num in range(numEpochs):
        if Epoch_Num > 1 and diff < Training_Diff:
            print("Step %d, change in cost %g; convergence." % (Epoch_Num, diff))
            break
        else:
            Train_Augmented_Matrix_View1_Minibatch, Train_Augmented_Matrix_View2_Minibatch, Train_Augmented_Matrix_View3_Minibatch, Train_Augmented_Label_Vector_Minibatch = next_batch(Minibatch_Size, Epoch_Num, Train_Augmented_Matrix_View1, Train_Augmented_Matrix_View2, Train_Augmented_Matrix_View3, Train_Augmented_Label_Vector)
            training_OP.run(feed_dict={X1: Train_Augmented_Matrix_View1_Minibatch, X2: Train_Augmented_Matrix_View2_Minibatch, X3: Train_Augmented_Matrix_View3_Minibatch, y_Gold: Train_Augmented_Label_Vector_Minibatch, is_training: True})
            if Epoch_Num % 10 == 0:
                Train_Cost = sess.run(Loss_fun, feed_dict={X1: Train_Augmented_Matrix_View1_Minibatch, X2: Train_Augmented_Matrix_View2_Minibatch, X3: Train_Augmented_Matrix_View3_Minibatch, y_Gold: Train_Augmented_Label_Vector_Minibatch, is_training: True})
                if np.isnan(Train_Cost):
                    break
                else:
                    newCost = Train_Cost
                    diff = abs(newCost - cost)
                    cost = newCost
                    print("Step %d: cost %g, change in cost %g" % (Epoch_Num, newCost, diff))

    # ***** Testing *****
    y_Gold_Test = Test_Label_Vector
    y_Deep_NN_Test = sess.run(y_Deep_NN, feed_dict={X1: Test_Matrix_View1, X2: Test_Matrix_View2, X3: Test_Matrix_View3, is_training: False})

    # ***** Save *****:
    a = {}
    a['Stage_' + str(test_stage) + '_y_Gold'] = y_Gold_Test
    a['Stage_' + str(test_stage) + '_y_Deep_NN'] = y_Deep_NN_Test

    a['Stage_' + str(test_stage) + '_Test_Benign_Masses_Num'] = Test_Benign_Masses_Num
    a['Stage_' + str(test_stage) + '_Train_Benign_Masses_Num'] = Train_Benign_Masses_Num

    a['Stage_' + str(test_stage) + '_Test_Malignant_Masses_Num'] = Test_Malignant_Masses_Num
    a['Stage_' + str(test_stage) + '_Train_Malignant_Masses_Num'] = Train_Malignant_Masses_Num

    a['randsampled_benign_masses'] = randsampled_benign_masses
    a['randsampled_malignant_masses'] = randsampled_malignant_masses

    a['splitting_benign_masses'] = splitting_benign_masses
    a['splitting_malignant_masses'] = splitting_malignant_masses

    Name = Address + '\Stage_' + str(test_stage) + '_y_Gold_y_Deep_NN_' + View_Name[0] + '_' + View_Name[1] + '_' + View_Name[2] + '.mat'
    savemat(Name, a)
    sess.close()
    return


def main_run_deep_model(View_Name, randsampled_benign_masses, randsampled_malignant_masses, splitting_benign_masses, splitting_malignant_masses, test_stage, Address):
    run_deep_model(View_Name, randsampled_benign_masses, randsampled_malignant_masses, splitting_benign_masses, splitting_malignant_masses, test_stage, Address)
    return


########################################################################################################################
#**************************************************** Initialization ***************************************************
masses_num_benign = 79
masses_num_malignant = 77
View_Names = np.array([['US', 'MG_CC', 'MG_MLO']])
k = 5 # k-fold cross validation
test_stage_1 = 1
test_stage_2 = k
Address = r'C:\Output/'

# ####################################################### Program ######################################################
benign_masses = list(range(1, masses_num_benign+1))
randsampled_benign_masses = sample(benign_masses,masses_num_benign) # Shuffling
splitting_benign_masses = splitting_cv(masses_num_benign,k) # (k, 1)

malignant_masses = list(range(1, masses_num_malignant+1))
randsampled_malignant_masses = sample(malignant_masses,masses_num_malignant) # Shuffling
splitting_malignant_masses = splitting_cv(masses_num_malignant,k) # (k, 1)

for View_Num in range(View_Names.shape[0]):
    for test_stage in range(test_stage_1, test_stage_2 + 1):
        run_deep_model(View_Names[View_Num], randsampled_benign_masses, randsampled_malignant_masses, splitting_benign_masses, splitting_malignant_masses, test_stage, Address)
        a = {}
        a = loadmat(Address + '\Stage_' + str(test_stage) + '_y_Gold_y_Deep_NN_' + View_Names[View_Num][0] + '_' + View_Names[View_Num][1] + '_' + View_Names[View_Num][2] + '.mat')
        y_Deep_NN_Check = a['Stage_' + str(test_stage) + '_y_Deep_NN']
        while np.isnan(y_Deep_NN_Check[0, 0]):
            main_run_deep_model(View_Names[View_Num], randsampled_benign_masses, randsampled_malignant_masses, splitting_benign_masses, splitting_malignant_masses, test_stage, Address)
            a = {}
            a = loadmat(Address + '\Stage_' + str(test_stage) + '_y_Gold_y_Deep_NN_' + View_Names[View_Num][0] + '_' + View_Names[View_Num][1] + '_' + View_Names[View_Num][2] + '.mat')
            y_Deep_NN_Check = a['Stage_' + str(test_stage) + '_y_Deep_NN']
    save_all(View_Names[View_Num], k, Address)
    evaluation(View_Names[View_Num], k, Address)