def model(X_Image1, X_Image2, X_Image3, update_ops_collection, is_training):
    # ***** ResNet50_3_4 ***** : Branch1x: [Start-Part3), Branch2y: [Part3-Part4), Branch3: [Part4-End]
    # ******************************************************************************************************************
    # Branch11: [Start-Part3) => (Input: X_Image1, Output: Addition3_res2_Branch11)
    MaxPool_Branch11 = part1(X_Image1, update_ops_collection, is_training)
    Addition3_res2_Branch11 = part2(MaxPool_Branch11, update_ops_collection, is_training)
    # Branch12: [Start-Part3) => (Input: X_Image2, Output: Addition3_res2_Branch12)
    MaxPool_Branch12 = part1(X_Image2, update_ops_collection, is_training)
    Addition3_res2_Branch12 = part2(MaxPool_Branch12, update_ops_collection, is_training)
    # Branch13: [Start-Part3) => (Input: X_Image1, Output: Addition3_res2_Branch13)
    MaxPool_Branch13 = part1(X_Image1, update_ops_collection, is_training)
    Addition3_res2_Branch13 = part2(MaxPool_Branch13, update_ops_collection, is_training)
    # Branch14: [Start-Part3) => (Input: X_Image3, Output: Addition3_res2_Branch14)
    MaxPool_Branch14 = part1(X_Image3, update_ops_collection, is_training)
    Addition3_res2_Branch14 = part2(MaxPool_Branch14, update_ops_collection, is_training)
    # Branch15: [Start-Part3) => (Input: X_Image2, Output: Addition3_res2_Branch15)
    MaxPool_Branch15 = part1(X_Image2, update_ops_collection, is_training)
    Addition3_res2_Branch15 = part2(MaxPool_Branch15, update_ops_collection, is_training)
    # Branch16: [Start-Part3) => (Input: X_Image3, Output: Addition3_res2_Branch16)
    MaxPool_Branch16 = part1(X_Image3, update_ops_collection, is_training)
    Addition3_res2_Branch16 = part2(MaxPool_Branch16, update_ops_collection, is_training)
    # ******************************************************************************************************************
    # Concatenation21 (Addition3_res2_Branch11, Addition3_res2_Branch12) => Conv_ReLU_New21
    Concatenation21 = tf.concat([Addition3_res2_Branch11, Addition3_res2_Branch12], 3)
    Conv_ReLU_New21 = tf.nn.relu(conv_layer_ago(Concatenation21, 1, 256, 1))  # (1×1, 256, S=1)
    # Concatenation22 (Addition3_res2_Branch13, Addition3_res2_Branch14) => Conv_ReLU_New22
    Concatenation22 = tf.concat([Addition3_res2_Branch13, Addition3_res2_Branch14], 3)
    Conv_ReLU_New22 = tf.nn.relu(conv_layer_ago(Concatenation22, 1, 256, 1))  # (1×1, 256, S=1)
    # Concatenation23 (Addition3_res2_Branch15, Addition3_res2_Branch16) => Conv_ReLU_New23
    Concatenation23 = tf.concat([Addition3_res2_Branch15, Addition3_res2_Branch16], 3)
    Conv_ReLU_New23 = tf.nn.relu(conv_layer_ago(Concatenation23, 1, 256, 1))  # (1×1, 256, S=1)
    # ******************************************************************************************************************
    # Branch21: [Part3-Part4) => (Input: Conv_ReLU_New21, Output: Addition4_res3_Branch21)
    Addition4_res3_Branch21 = part3(Conv_ReLU_New21, update_ops_collection, is_training)
    # Branch22: [Part3-Part4) => (Input: Conv_ReLU_New22, Output: Addition4_res3_Branch22)
    Addition4_res3_Branch22 = part3(Conv_ReLU_New22, update_ops_collection, is_training)
    # Branch23: [Part3-Part4) => (Input: Conv_ReLU_New23, Output: Addition4_res3_Branch23)
    Addition4_res3_Branch23 = part3(Conv_ReLU_New23, update_ops_collection, is_training)
    # ******************************************************************************************************************
    # Concatenation (Addition4_res3_Branch21, Addition4_res3_Branch22, Addition4_res3_Branch23) => Conv_ReLU_New
    Concatenation = tf.concat([Addition4_res3_Branch21, Addition4_res3_Branch22, Addition4_res3_Branch23], 3)
    Conv_ReLU_New = tf.nn.relu(conv_layer_ago(Concatenation, 1, 512, 1))  # (1×1, 512, S=1)
    # ******************************************************************************************************************
    # Branch3: [Part4-End] => (Input: Conv_ReLU_New, Output: Before_Softmax)
    Addition6_res4_Branch3 = part4(Conv_ReLU_New, update_ops_collection, is_training)
    Before_Softmax = part5(Addition6_res4_Branch3, update_ops_collection, is_training)
    return Before_Softmax