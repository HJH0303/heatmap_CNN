import numpy as np
from data_load_12 import Data_Pre

def pixel_to_table_cell(pixel_position, top_left_corner, col_size, row_size):
    rows, cols = 4, 4
    bottom_right_corner = (top_left_corner[0] + cols * col_size, top_left_corner[1] + rows * row_size)

    # Check if the pixel is outside the table area
    if not (top_left_corner[0] <= pixel_position[0] < bottom_right_corner[0] and
            top_left_corner[1] <= pixel_position[1] < bottom_right_corner[1]):
        return (-1, -1)

    # Calculate the row and column based on the pixel position
    col = (pixel_position[0] - top_left_corner[0]) // col_size
    row = (pixel_position[1] - top_left_corner[1]) // row_size
    
    # Check if the position is in one of the removed corners
    if (row == 0 and col in [0, 3]) or (row == 3 and col in [0, 3]):
        return (-1, -1)

    # Adjust the col value for the skipped columns in the top and bottom rows
    if row in [0, 3] and col > 0:
        col -= 1

    return (row, col)

def row_col_to_class(row, col):
    """
    Convert grid position (row, col) to a class index in a 4x4 grid with corners removed.
    
    Args:
    - row: row index in the grid
    - col: col index in the grid
    
    Returns:
    - class_index: The class index corresponding to the (row, col) in the modified grid
    """
    
    # Mapping for the classes, skipping the corners
    if row == 0:
        class_index = col + 1
    elif row == 1:
        class_index = 2 + col + 1
    elif row == 2:
        class_index = 6 + col + 1
    elif row == 3:
        class_index = 10 + col

    return class_index

if __name__ == "__main__":
    col_size =59
    row_size = 50
    top_left_corner = (260, 140)
    rows, cols = 4, 4
    input_arr, input_label = Data_Pre.data_load()
    output_array = np.zeros((input_label.shape[0], 1), dtype=int)  
    cell_counts = np.zeros((rows, cols), dtype=int)
  
    for idx, i in enumerate(input_label):
        x = i[0]
        y = i[1]
        row, col = pixel_to_table_cell((x, y), top_left_corner, col_size, row_size)
        # Ensure row and col are within the valid range
        if 0 <= row < rows and 0 <= col < cols:
            class_number = row_col_to_class(row, col)
            if class_number == 16: print("hi")
            output_array[idx] = class_number
            cell_counts[int(row), int(col)] += 1

    np.save(f"/home/aims/2024/dataset/Heat_map_Dataset/grid_label/2input_label_12",output_array)
    print(output_array.shape)