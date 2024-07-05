import cv2
import numpy as np

def filter(src_path, clean_path, dst_path):
    """
    Load image in 'src_path', perform this function,
    and then save your result image to 'dst_path'.
    """
    noisy_img = cv2.imread(src_path)
    clean_img = cv2.imread(clean_path)
    result_img = None

    """
    [Usage]
    result_img = median_filter(noisy_img)
    result_img = bilateral_filter(noisy_img)
    """
    result_img = bilateral_filter(noisy_img)
    
    rmse = calculate_rmse(result_img, clean_img)

    cv2.imwrite(dst_path, result_img)
    
    return rmse


# Median filter

def median_filter(img):
    
    kernel_size = 3
    edge = int((kernel_size - 1) / 2)

    rows = len(img)
    cols = len(img[0])

    img_result = np.full((rows, cols, 3), 0)

    for row in range(rows):
        for col in range(cols):
            temp_0, temp_1, temp_2 = [], [], []

            for result_row in range(row-edge, row+edge+1):
                for result_col in range(col-edge, col+edge+1):
                    try:
                        temp_0.append(img[result_row, result_col, 0])
                    except:
                        pass
                    try:
                        temp_1.append(img[result_row, result_col, 1])
                    except:
                        pass
                    try:
                        temp_2.append(img[result_row, result_col, 2])
                    except:
                        pass

            color_0, color_1, color_2 = np.median(
                temp_0), np.median(temp_1), np.median(temp_2)
            img_result[row, col] = [color_0, color_1, color_2]
            
    img = img_result
    
    return img


# Bilateral filter

def gaussian_kernel(x, sigma):
    
    return np.exp(- (x ** 2) / (2 * sigma ** 2))


def apply_bilateral_filter(image, kernel_size, sigma_i, sigma_s):
    half_kernel_size = kernel_size // 2

    H, W, C = image.shape

    padded_image = np.zeros((H +2 * half_kernel_size, W + 2 * half_kernel_size, C), dtype=np.float64)
    padded_image[half_kernel_size:H+half_kernel_size, half_kernel_size:W+half_kernel_size, :] = image.astype(np.float64)
    
    for c in range(C):
        for h in range(H):
            for w in range(W):
                weight_sum = 0
                filtered_value = 0
                for i in range(-half_kernel_size, half_kernel_size+1):
                    for j in range(-half_kernel_size, half_kernel_size+1):
                        window_h = h + i
                        window_w = w + j
                        gi = gaussian_kernel(padded_image[half_kernel_size+window_h][half_kernel_size+window_w][c] - padded_image[half_kernel_size+h][half_kernel_size+w][c], sigma_i)
                        gs = gaussian_kernel(np.sqrt(i**2 + j**2), sigma_s)
                        weight = gi * gs
                        filtered_value += padded_image[half_kernel_size+window_h][half_kernel_size+window_w][c] * weight
                        weight_sum += weight
                image[h][w][c] = filtered_value / weight_sum
    
    filtered_image = image

    return filtered_image




def bilateral_filter(img):

    img = apply_bilateral_filter(img, 5, 50, 4)
    
    return img


def calculate_rmse(img1, img2):
    """
    Calculates RMS error between two images. Two images should have same sizes.
    """
    if (img1.shape[0] != img2.shape[0]) or \
            (img1.shape[1] != img2.shape[1]) or \
            (img1.shape[2] != img2.shape[2]):
        raise Exception("img1 and img2 should have sime sizes.")

    diff = np.abs(img1.astype(dtype=int) - img2.astype(dtype=int))
    return np.sqrt(np.mean(diff ** 2))


if __name__ == "__main__":
    # Define paths for the source, clean, and destination images
    src_path = "src/fox_noisy.jpg"
    clean_path = "src/fox_clean.jpg"
    dst_path = "src/fox_result.jpg"
      
    # Execute the main function for task 3
    rmse = filter(src_path, clean_path, dst_path)
    
    # Output the RMSE to evaluate the noise removal effectiveness
    print(f"RMSE: {rmse}")
