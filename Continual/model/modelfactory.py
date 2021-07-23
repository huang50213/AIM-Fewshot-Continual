class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type, dataset, in_channels=6, num_actions=6, width=300):
        if dataset == "omniglot":
            nm_channels = 112
            channels = 256
            size_of_representation = 2304
            size_of_interpreter = 1008
            if model_type == "ANML+AIM":
                nm_channels = 112
                channels = 256
                size_of_representation = 2304
                size_of_interpreter = 1008
                return [
                    # =============== Separate network neuromodulation =======================
                    ('conv1_nm', [nm_channels, 3, 3, 3, 1, 0]),
                    ('bn1_nm', [nm_channels]),
                    ('conv2_nm', [nm_channels, nm_channels, 3, 3, 1, 0]),
                    ('bn2_nm', [nm_channels]),
                    ('conv3_nm', [nm_channels, nm_channels, 3, 3, 1, 0]),
                    ('bn3_nm', [nm_channels]),
                    ('nm_to_fc', [size_of_representation, size_of_interpreter]),
                    # =============== Prediction network ===============================
                    ('conv1', [channels, 3, 3, 3, 1, 0]),
                    ('bn1', [channels]),
                    ('conv2', [channels, channels, 3, 3, 1, 0]),
                    ('bn2', [channels]),
                    ('conv3', [channels, channels, 3, 3, 1, 0]),
                    ('bn3', [channels]),
                    ('fc', [1000, size_of_representation // 2]),
                    ('linear', [size_of_representation // 2, size_of_representation]),
                    ('aim', [size_of_representation // 2, 128, 64, 128, size_of_representation // 2, 128])
                ]
            elif model_type == "OML+AIM":
                return [
                    # =============== slow weight =======================
                    ('conv1_nm', [nm_channels, 3, 3, 3, 1, 0]),
                    ('bn1_nm', [nm_channels]),
                    ('conv2_nm', [nm_channels, nm_channels, 3, 3, 1, 0]),
                    ('bn2_nm', [nm_channels]),
                    ('conv3_nm', [nm_channels, nm_channels, 3, 3, 1, 0]),
                    ('bn3_nm', [nm_channels]),
                    ('conv4_nm', [nm_channels, nm_channels, 3, 3, 1, 0]),
                    ('bn4_nm', [nm_channels]),
                    ('conv5_nm', [nm_channels, nm_channels, 3, 3, 1, 0]),
                    ('bn5_nm', [nm_channels]),
                    ('conv6_nm', [nm_channels, nm_channels, 3, 3, 1, 0]),
                    ('bn6_nm', [nm_channels]),
                    ('nm_to_fc', [size_of_interpreter // 2, size_of_interpreter]),
                    # =============== fast weight =======================
                    ('fc', [1000, size_of_interpreter // 2]),
                    ('aim', [size_of_interpreter // 2, 128, 64, 128, size_of_interpreter // 2, 128]),
                    # [input_size, hidden_size, num_units, input_key_size, input_value_size, input_query_size]
                ]
        elif dataset == "cifar100":
            nm_channels = 112
            channels = 256
            size_of_representation = 1024
            size_of_interpreter = 1792
            if model_type == "ANML+AIM":
                size_of_representation = 4096
                size_of_interpreter = 1792
                return [
                    # =============== slow weight =======================
                    ('conv1_nm', [nm_channels, 3, 3, 3, 1, 0]),
                    ('bn1_nm', [nm_channels]),
                    ('conv2_nm', [nm_channels, nm_channels, 3, 3, 1, 0]),
                    ('bn2_nm', [nm_channels]),
                    ('conv3_nm', [nm_channels, nm_channels, 3, 3, 1, 0]),
                    ('bn3_nm', [nm_channels]),
                    ('nm_to_fc', [size_of_representation, size_of_interpreter]),
                    # =============== fast weight =======================
                    ('conv1', [channels, 3, 3, 3, 1, 0]),
                    ('bn1', [channels]),
                    ('conv2', [channels, channels, 3, 3, 1, 0]),
                    ('bn2', [channels]),
                    ('conv3', [channels, channels, 3, 3, 1, 0]),
                    ('bn3', [channels]),
                    ('fc', [100, size_of_representation // 4]),
                    ('linear', [size_of_representation // 4, size_of_representation]),
                    ('aim', [size_of_representation // 4, 128, 64, 128, size_of_representation // 4, 128]),
                ]
            elif model_type == "OML+AIM":
                return [
                    # =============== slow weight =======================
                    ('conv1_nm', [nm_channels, 3, 3, 3, 1, 0]),
                    ('bn1_nm', [nm_channels]),
                    ('conv2_nm', [nm_channels, nm_channels, 3, 3, 1, 0]),
                    ('bn2_nm', [nm_channels]),
                    ('conv3_nm', [nm_channels, nm_channels, 3, 3, 1, 0]),
                    ('bn3_nm', [nm_channels]),
                    ('conv4_nm', [nm_channels, nm_channels, 3, 3, 1, 0]),
                    ('bn4_nm', [nm_channels]),
                    ('conv5_nm', [nm_channels, nm_channels, 3, 3, 1, 0]),
                    ('bn5_nm', [nm_channels]),
                    ('conv6_nm', [nm_channels, nm_channels, 3, 3, 1, 0]),
                    ('bn6_nm', [nm_channels]),
                    ('nm_to_fc', [size_of_interpreter // 2, size_of_interpreter]),
                    # =============== fast weight =======================
                    ('fc', [100, size_of_interpreter // 2]),
                    ('aim', [size_of_interpreter // 2, 128, 64, 128, size_of_interpreter // 2, 128]),
                    # [input_size, hidden_size, num_units, input_key_size, input_value_size, input_query_size]
                ]
        elif dataset == "imagenet":
            nm_channels = 112
            channels = 256
            size_of_representation = 2304
            size_of_interpreter = 2800
            if model_type == "ANML+AIM":
                size_of_representation = 16384
                size_of_interpreter = 7168
                return [
                    # =============== slow weight =======================
                    ('conv1_nm', [nm_channels, 3, 3, 3, 1, 0]),
                    ('bn1_nm', [nm_channels]),
                    ('conv2_nm', [nm_channels, nm_channels, 3, 3, 1, 0]),
                    ('bn2_nm', [nm_channels]),
                    ('conv3_nm', [nm_channels, nm_channels, 3, 3, 1, 0]),
                    ('bn3_nm', [nm_channels]),
                    ('nm_to_fc', [size_of_representation, size_of_interpreter]),
                    # =============== fast weight =======================
                    ('conv1', [channels, 3, 3, 3, 1, 0]),
                    ('bn1', [channels]),
                    ('conv2', [channels, channels, 3, 3, 1, 0]),
                    ('bn2', [channels]),
                    ('conv3', [channels, channels, 3, 3, 1, 0]),
                    ('bn3', [channels]),
                    ('fc', [84, size_of_representation // 16]),
                    ('linear', [size_of_representation // 16, size_of_representation]),
                    ('aim', [size_of_representation // 16, 128, 64, 128, size_of_representation // 16, 128]),
                ]
            elif model_type == "OML+AIM":
                return [
                    # =============== slow weight =======================
                    ('conv1_nm', [nm_channels, 3, 3, 3, 1, 0]),
                    ('bn1_nm', [nm_channels]),
                    ('conv2_nm', [nm_channels, nm_channels, 3, 3, 1, 0]),
                    ('bn2_nm', [nm_channels]),
                    ('conv3_nm', [nm_channels, nm_channels, 3, 3, 1, 0]),
                    ('bn3_nm', [nm_channels]),
                    ('conv4_nm', [nm_channels, nm_channels, 3, 3, 1, 0]),
                    ('bn4_nm', [nm_channels]),
                    ('conv5_nm', [nm_channels, nm_channels, 3, 3, 1, 0]),
                    ('bn5_nm', [nm_channels]),
                    ('conv6_nm', [nm_channels, nm_channels, 3, 3, 1, 0]),
                    ('bn6_nm', [nm_channels]),
                    ('nm_to_fc', [size_of_interpreter // 2, size_of_interpreter]),
                    # =============== fast weight =======================
                    ('fc', [84, size_of_interpreter // 2]),
                    ('aim', [size_of_interpreter // 2, 128, 32, 128, size_of_interpreter // 2, 128]),
                    # [input_size, hidden_size, num_units, input_key_size, input_value_size, input_query_size]
                ]
        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)
