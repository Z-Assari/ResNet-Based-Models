def model(X_Image1, X_Image2, X_Image3, update_ops_collection, is_training):
    # ***** ResNet50_2_5 ***** : Branch1x: [Start-Part2), Branch2y: [Part2-Part5), Branch3: [Part5-End]
    # ******************************************************************************************************************
    # Branch11: [Start-Part2) => (Input: X_Image1, Output: MaxPool_Branch11)
    MaxPool_Branch11 = part1(X_Image1, update_ops_collection, is_training)
    # Branch12: [Start-Part2) => (Input: X_Image2, Output: MaxPool_Branch12)
    MaxPool_Branch12 = part1(X_Image2, update_ops_collection, is_training)
    # Branch13: [Start-Part2) => (Input: X_Image1, Output: MaxPool_Branch13)
    MaxPool_Branch13 = part1(X_Image1, update_ops_collection, is_training)
    # Branch14: [Start-Part2) => (Input: X_Image3, Output: MaxPool_Branch14)
    MaxPool_Branch14 = part1(X_Image3, update_ops_collection, is_training)
    # Branch15: [Start-Part2) => (Input: X_Image2, Output: MaxPool_Branch15)
    MaxPool_Branch15 = part1(X_Image2, update_ops_collection, is_training)
    # Branch16: [Start-Part2) => (Input: X_Image3, Output: MaxPool_Branch16)
    MaxPool_Branch16 = part1(X_Image3, update_ops_collection, is_training)
    # ******************************************************************************************************************
    # Concatenation21 (MaxPool_Branch11, MaxPool_Branch12) => Conv_ReLU_New21
    Concatenation21 = tf.concat([MaxPool_Branch11, MaxPool_Branch12], 3)
    Conv_ReLU_New21 = tf.nn.relu(conv_layer_ago(Concatenation21, 7, 64, 1))  # (7×7, 64, S=1)
    # Concatenation22 (MaxPool_Branch13, MaxPool_Branch14) => Conv_ReLU_New22
    Concatenation22 = tf.concat([MaxPool_Branch13, MaxPool_Branch14], 3)
    Conv_ReLU_New22 = tf.nn.relu(conv_layer_ago(Concatenation22, 7, 64, 1))  # (7×7, 64, S=1)
    # Concatenation23 (MaxPool_Branch15, MaxPool_Branch16) => Conv_ReLU_New23
    Concatenation23 = tf.concat([MaxPool_Branch15, MaxPool_Branch16], 3)
    Conv_ReLU_New23 = tf.nn.relu(conv_layer_ago(Concatenation23, 7, 64, 1))  # (7×7, 64, S=1)
    # ******************************************************************************************************************
    # Branch21: [Part2-Part5) => (Input: Conv_ReLU_New21, Output: Addition6_res4_Branch21)
    Addition3_res2_Branch21 = part2(Conv_ReLU_New21, update_ops_collection, is_training)
    Addition4_res3_Branch21 = part3(Addition3_res2_Branch21, update_ops_collection, is_training)
    Addition6_res4_Branch21 = part4(Addition4_res3_Branch21, update_ops_collection, is_training)
    # Branch22: [Part2-Part5) => (Input: Conv_ReLU_New22, Output: Addition6_res4_Branch22)
    Addition3_res2_Branch22 = part2(Conv_ReLU_New22, update_ops_collection, is_training)
    Addition4_res3_Branch22 = part3(Addition3_res2_Branch22, update_ops_collection, is_training)
    Addition6_res4_Branch22 = part4(Addition4_res3_Branch22, update_ops_collection, is_training)
    # Branch23: [Part2-Part5) => (Input: Conv_ReLU_New23, Output: Addition6_res4_Branch23)
    Addition3_res2_Branch23 = part2(Conv_ReLU_New23, update_ops_collection, is_training)
    Addition4_res3_Branch23 = part3(Addition3_res2_Branch23, update_ops_collection, is_training)
    Addition6_res4_Branch23 = part4(Addition4_res3_Branch23, update_ops_collection, is_training)
    # ******************************************************************************************************************
    # Concatenation (Addition6_res4_Branch21, Addition6_res4_Branch22, Addition6_res4_Branch23) => Conv_ReLU_New
    Concatenation = tf.concat([Addition6_res4_Branch21, Addition6_res4_Branch22, Addition6_res4_Branch23], 3)
    Conv_ReLU_New = tf.nn.relu(conv_layer_ago(Concatenation, 1, 1024, 1))  # (1×1, 1024, S=1)
    # ******************************************************************************************************************
    # Branch3: [Part5-End] => (Input: Conv_ReLU_New, Output: Before_Softmax)
    Before_Softmax = part5(Conv_ReLU_New, update_ops_collection, is_training)
    return Before_Softmax