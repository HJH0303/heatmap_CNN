import numpy as np

class Data_Pre:
    # ------데이터 load-------
    def data_load():
        input_arr=np.load(f"/home/aims/2024/dataset/Heat_map_Dataset/input_arr/input_arr_1.npy")
        num_video=2
        last_video=21
        while True:
            temp_arr=np.load(f"/home/aims/2024/dataset/Heat_map_Dataset/input_arr/input_arr_{num_video}.npy")
            input_arr=np.concatenate((input_arr,temp_arr),axis=0)
            if num_video==last_video:
                break
            else:
                num_video+=1
        input_label=np.load("/home/aims/2024/dataset/Heat_map_Dataset/grid_label/2input_label_4x4.npy")
        shape=input_arr.shape
        shape2=input_label.shape

        print(shape)
        print(shape2)

        return input_arr, input_label

