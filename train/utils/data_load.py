import numpy as np

class Data_Pre:
    # ------데이터 load-------
    def data_load():
        input_arr = np.load(f"/home/aims/2024/dataset/Heat_map_Dataset/input_arr/input_arr_1.npy")
        input_label =np.load(f"/home/aims/2024/dataset/Heat_map_Dataset/input_label/input_label_1.npy")

        num_video=1 
        last_video=21
        while True:
            temp_arr=np.load(f"/home/aims/2024/dataset/Heat_map_Dataset/input_arr/input_arr_{num_video}.npy")
            temp_label=np.load(f"/home/aims/2024/dataset/Heat_map_Dataset/input_label/input_label_{num_video}.npy")

            input_arr=np.concatenate((input_arr,temp_arr),axis=0)
            input_label=np.concatenate((input_label,temp_label),axis=0)

            if num_video==last_video:
                break
            else:
                num_video+=1
        shape=input_arr.shape
        shape2=input_label.shape

        print(shape)
        print(shape2)

        return input_arr, input_label

if __name__ == "__main__":
    Data_Pre.data_load()