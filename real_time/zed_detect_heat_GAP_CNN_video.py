from ultralytics import YOLO
import numpy as np
import cv2
import torch
from torch import nn
from collections import defaultdict
from scipy.ndimage import gaussian_filter
from my_model.my_model_CNN_heat_GAP import CNN_heat
from utils.data_pre import real_time_data_pre
np.set_printoptions(threshold=np.inf, linewidth=np.inf) 
def draw_table_on_image(img, top_left_corner, cell_size):
    # Load the image
    # Calculate the bottom right corner of the table based on the top left corner and cell size
    rows, cols = 4, 4
    bottom_right_corner = (top_left_corner[0] + cols * cell_size, top_left_corner[1] + rows * cell_size)

    # Draw the horizontal lines
    for row in range(rows + 1):
        start_point = (top_left_corner[0], top_left_corner[1] + row * cell_size)
        end_point = (bottom_right_corner[0], top_left_corner[1] + row * cell_size)
        img = cv2.line(img, start_point, end_point, (0, 0, 255), 2)

    # Draw the vertical lines
    for col in range(cols + 1):
        start_point = (top_left_corner[0] + col * cell_size, top_left_corner[1])
        end_point = (top_left_corner[0] + col * cell_size, bottom_right_corner[1])
        img = cv2.line(img, start_point, end_point, (0, 0, 255), 2)
    return img
def remove_module_prefix(state_dict):
    """저장된 모델의 state_dict에서 'module.' 접두사를 제거합니다."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            # 'module.' 제거
            new_key = k[7:]
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict

    return img
def main():
    class_arr = [(267, 157), (322, 157), (377, 157), (432, 157),
                 (267, 212), (322, 212), (377, 212), (432, 212),
                 (267, 267), (322, 267), (377, 267), (432, 267),
                 (267, 322), (322, 322), (377, 322), (432, 322)]
    # parameters
    PATH_weight="/home/aims/2024/weights/heatmap/gap/best/140.pth"
    DEVICE = '0'
    blue_color = (255, 0, 0)
    red_color = (0, 0, 255)
    center_axis=340
    # Initialize
    print('device:', DEVICE)

    # Load model 
    # v8_segment
    model_v8= YOLO('/home/aims/obb_contents/weights/v8/best_04_10_ nano.pt') 

    # gait_cycle_model
    model_heat = CNN_heat().to(device)
    # 체크포인트 로드
    checkpoint = torch.load(PATH_weight)


    # 'module.' 접두사 제거
    updated_state_dict = remove_module_prefix(checkpoint)



    model_heat.load_state_dict(updated_state_dict)
    # model_heat.load_state_dict(torch.load(PATH_weight))
    model_heat.eval()
    

    # if half:
    #     model_obb.model.half()  # to FP16

    print("Running...")
    # zed parameter setting
    # cap = cv2.VideoCapture("/home/aims/yolov8_8_0_10/ultralytics/yolov5_obb/video/start_confused_foot.avi")

    # video load
    # cap = cv2.VideoCapture("/home/aims/2024/dataset/video/2024_05_02/test/output_7.avi")
    cap = cv2.VideoCapture("/home/aims/2024/dataset/video/2024_05_16/test_2.avi")
    # cap = cv2.VideoCapture("/home/aims/obb_contents/no_rock/video5.avi")

    # heatmap 초기화
    track_history = defaultdict(lambda: [])
    track = []
    softmax = nn.Softmax(dim=None)

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay=30
    cell_size = 70
    img_size_cell = 4 * cell_size
    real_data=real_time_data_pre()
    key = ''    
    num_loo=0
    with torch.no_grad():
        
        while key != 113:  # for 'q' key
            ret, frame =cap.read()
            # print(ret)
            if ret:
                temp_img=frame.copy()
                results = model_v8.track(frame, persist=True)
                boxes = results[0].boxes.xywh.cpu()
                heatmap = np.zeros(( int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))), dtype=np.float32)
                prev_track = track
                rocks,j_r,foots,j_f,mask_r  ,mask_f =real_time_data_pre.get_info(results)
                # temp_img=real_data.draw_rect(temp_img,rocks,foots,j_r,j_f)
                real_data.data_foot(foots,j_f,real_data)
                if results[0].boxes.id == None or np.all(results[0].boxes.cls.detach().cpu().numpy() == 0) : 
                    for (x, y) in prev_track:
                        heatmap[y:y+2, x:x+2] += 1  # Simple increment
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
                        for (x, y) in reversed(track):
                            heatmap[y:y+2, x:x+2] += 1  # Simple increment
                    
                    smoothed_heatmap = gaussian_filter(heatmap, sigma=10)
                    heatmap_img = np.uint8(255 * smoothed_heatmap / np.max(smoothed_heatmap))
                    heatmap_img = np.reshape(heatmap_img,(376,672,1))
                    # Convert heatmap to color
                    heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)

                    # Overlay the heatmap on the annotated frame
                    # cv2.addWeighted(heatmap_color, 0.6, temp_img, 0.4, 0, temp_img)
                # 채널 병합
                image_array = np.array(frame)
                # 차원 추가
                print(heatmap_img.shape)
                extended_image = np.concatenate((image_array, heatmap_img), axis=2)
                start_x, start_y = 240, 116  # crop된 이미지의 시작점
                end_x, end_y = 500, 376  # crop된 이미지의 끝점
                # 이미지를 원하는 크기로 자릅니다.
                cropped_image = extended_image[start_y:end_y, start_x:end_x]
                rocks,j_r,foots,j_f,mask_r  ,mask_f =real_time_data_pre.get_info(results)
                cropped_image = torch.from_numpy(cropped_image).to(device).float()
                cropped_image = cropped_image.unsqueeze(0)
                cropped_image = cropped_image.permute(0, 3, 1, 2)
                outputs = model_heat(cropped_image)
                _, predicted = torch.max(outputs, 1)

                # Grid visualization 
                soft_output = softmax(outputs)
                matrix_44 = soft_output.view(4, 4).cpu()
                matrix_44 = (matrix_44 * 255).int().numpy()
                
                visualization = np.zeros((img_size_cell, img_size_cell, 3), dtype=np.uint8)
                for i in range(4):
                    for j in range(4):
                        # 색상 결정: 확률 값에 따라 grayscale로 적용
                        color = matrix_44[i, j].item()
                        cv2.rectangle(visualization, (j * cell_size, i * cell_size),
                                    ((j + 1) * cell_size - 1, (i + 1) * cell_size - 1),
                                    (color, color, color), -1)

                cv2.imshow('Probability Visualization', visualization)
                temp_img=real_data.draw_rect(temp_img,rocks,foots,j_r,j_f)
                # temp_img=draw_table_on_image(temp_img, (240, 130), 55)
                temp_img=cv2.circle(temp_img, class_arr[int(predicted[0])], 5, blue_color, -1)
                # temp_img=cv2.line(temp_img,(center_axis,0),(center_axis,400),color=red_color,thickness=1)
                # input_data_check
                # temp_img=real_data.input_data_check(real_data,temp_img)
                cv2.imshow("ZED", temp_img)
                key = cv2.waitKey(delay)
                num_loo+=1
            else:
                break
    cv2.destroyAllWindows()
    cap.release()
    print("\nFINISH")

if __name__=="__main__":
    main()
