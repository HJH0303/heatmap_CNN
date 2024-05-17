import numpy as np
import cv2
from data_load import Data_Pre

def pixel_to_table_cell(pixel_position, top_left_corner, cell_size):
    rows, cols = 4, 4
    bottom_right_corner = (top_left_corner[0] + cols * cell_size, top_left_corner[1] + rows * cell_size)

    # Check if the pixel is outside the table area
    if not (top_left_corner[0] <= pixel_position[0] <= bottom_right_corner[0] and
            top_left_corner[1] <= pixel_position[1] <= bottom_right_corner[1]):
        return (-1, -1)

    # Calculate the row and column based on the pixel position
    col = (pixel_position[0] - top_left_corner[0]) // cell_size
    row = (pixel_position[1] - top_left_corner[1]) // cell_size
    
    return (row, col)

def row_col_to_class(row, col, cols=4): return row * cols + col
if __name__ == "__main__":
    cell_size = 55
    top_left_corner = (240, 130)
    rows, cols = 4, 4
    input_arr, input_label = Data_Pre.data_load()
    output_array = np.zeros((input_label.shape[0], 1), dtype=int)  
    cell_counts = np.zeros((rows, cols), dtype=int)
  
    for idx, i in enumerate(input_label):
        x = i[0]
        y = i[1]
        row, col = pixel_to_table_cell((x, y), top_left_corner, cell_size)
        # Ensure row and col are within the valid range
        if 0 <= row < rows and 0 <= col < cols:
            class_number = row_col_to_class(row, col)
            if class_number == 16: print("hi")
            output_array[idx] = class_number
            cell_counts[int(row), int(col)] += 1

    np.save(f"/home/aims/2024/dataset/Heat_map_Dataset/grid_label/2input_label_4x4",output_array)
    print(output_array.shape)