from ultralytics import YOLO
import numpy as np
import cv2
import torch
from data_pre import real_time_data_pre
from collections import defaultdict
from scipy.ndimage import gaussian_filter
import sys
epsilon = sys.float_info.epsilon
"""
RGB + heatmap 채널 만들기
영상별 (n,260,260,4) npy 파일 저장
"""
def main():
    # Initialize
    for ii in range(1,22):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device:', device)

        # v8 Load model 
        model_v8= YOLO('/home/aims/obb_contents/weights/v8/best_04_10_ nano.pt') 
        
        # video load
        # cap = cv2.VideoCapture("/home/aims/obb_contents/rock/test_rock1.avi")
        # circle
        num_video = ii
        start_file_arr=[48,   24,  39,  44,  25,  24,  27,  41,  37,  33,  31,  39, 62,  34,  34,  33,  31,  37,  34,  19,  28] 
        end_file_arr=  [283, 314, 269, 241, 205, 247, 241, 230, 231, 252, 236, 256, 302, 289, 262, 256, 234, 261, 250, 246, 289] 
        start_file= start_file_arr[num_video-1]
        end_file= end_file_arr[num_video-1]
        cap = cv2.VideoCapture(f"/home/aims/2024/dataset/video/2024_05_02/output_{num_video}.avi")
        # cap = cv2.VideoCapture("/home/aims/obb_contents/no_rock/video5.avi")

        # heatmap 초기화
        track_history = defaultdict(lambda: [])
        track = []
        # 임시저장할 이미지 list
        image_list = []
        delay=30
        key = ''    
        num_frame=1
        real_data=real_time_data_pre()

        with torch.no_grad():
            
            while key != 113:  # for 'q' key
                ret, frame =cap.read()
                # print(ret)
                if ret:
                    temp_img=frame.copy()
                    heatmap = np.full((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))), fill_value=epsilon, dtype=np.float32)

                    prev_track = track
                    # results = model_v8.track(source=frame, show=True)
                    results = model_v8.track(frame, persist=True,show=True,tracker='./param/custom_tracker.yaml')
                    boxes = results[0].boxes.xywh.cpu()
                    # 발 class가 아예 안잡혔을때는 이전 tracking 데이터를 사용한다.
                    if results[0].boxes.id == None or np.all(results[0].boxes.cls.detach().cpu().numpy() == 0) : 
                        for idx,(x, y) in enumerate(reversed(prev_track)):
                            heatmap[y:y+2, x:x+2] += 1/(idx+1)  # Simple increment
                        smoothed_heatmap = gaussian_filter(heatmap, sigma=10)
                        heatmap_img = np.uint8(255 * smoothed_heatmap / np.max(smoothed_heatmap))
                        heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
                        heatmap_img = np.reshape(heatmap_img,(376,672,1))
                    else:
                        track_ids = results[0].boxes.id.int().cpu().tolist()
                        cls = results[0].boxes.cls

                        for box, track_id,cl in zip(boxes, track_ids, cls):
                            if cl == 0 : continue # 바위 클래스면 heatmap 제외
                            x, y, w, h = box
                            track = track_history[track_id]
                            track.append((int(x), int(y)))  # x, y center point
                            if len(track) > 30:  # retain 30 tracks for 30 frames
                                track.pop(0)
                                # Update heatmap from the tracks
                            for idx,(x, y) in enumerate(reversed(track)):
                                heatmap[y:y+2, x:x+2] += 1/(idx+1)  # Simple increment
                        
                        smoothed_heatmap = gaussian_filter(heatmap, sigma=10)
                        heatmap_img = np.uint8(255 * smoothed_heatmap / np.max(smoothed_heatmap))
                        heatmap_img = np.reshape(heatmap_img,(376,672,1))
                    # Convert heatmap to color
                    heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)

                    # Overlay the heatmap on the annotated frame
                    cv2.addWeighted(heatmap_color, 0.6, temp_img, 0.4, 0, temp_img)
                    # 채널 병합
                    new_dimension = 4
                    image_array = np.array(frame)
                    # 차원 추가
                    print(heatmap_img.shape)
                    extended_image = np.concatenate((image_array, heatmap_img), axis=2)
                    start_x, start_y = 240, 116  # crop된 이미지의 시작점
                    end_x, end_y = 500, 376  # crop된 이미지의 끝점
                    # 이미지를 원하는 크기로 자릅니다.
                    cropped_image = extended_image[start_y:end_y, start_x:end_x]
                    if num_frame>=start_file and num_frame<=end_file: 
                        image_list.append(cropped_image)
                    rocks,j_r,foots,j_f,mask_r  ,mask_f =real_time_data_pre.get_info(results)
                    temp_img=real_data.draw_rect(temp_img,rocks,foots,j_r,j_f)

                    cv2.imshow("ZED", temp_img)
                    key = cv2.waitKey(delay)
                    num_frame += 1
                else:
                    break
        
        print("\nFINISH")
        images_array = np.stack(image_list, axis=0)
        np.save(file=f"/home/aims/2024/dataset/Heat_map_Dataset/input_arr/not_0/input_arr_{num_video}.npy",arr=images_array)
        rand_arr = np.random.randint(1,180, size = 3)
        # cv2.imshow("arr_0_img",images_array[rand_arr[0],:,:,0:3])
        # cv2.imshow("arr_1_img",images_array[rand_arr[1],:,:,0:3])
        # cv2.imshow("arr_2_img",images_array[rand_arr[2],:,:,0:3])

        # cv2.imshow("arr_0",images_array[rand_arr[0],:,:,3])
        # cv2.imshow("arr_1",images_array[rand_arr[1],:,:,3])
        # cv2.imshow("arr_2",images_array[rand_arr[2],:,:,3])
        # cv2.waitKey()
        cv2.destroyAllWindows()
        # cap.release()
        print(images_array.shape)
    

if __name__=="__main__":
    main()