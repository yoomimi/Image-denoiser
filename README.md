# ✨ Image-denoiser


### [1] Median filter
The median filter is a non-linear filtering technique used to remove noise from images, particularly effective for "salt and pepper" noise. It works as follows:

1. For each pixel in the image, define a kernel (or window) of a specified size centered on that pixel.
2. Sort all the pixel values within the kernel.
3. Select the median value from the sorted list of pixel values.
4. Replace the central pixel with this median value.


<br>

__RMSE = 7.87 with cat image (kernel size = 3)__

| ![Image 1](https://github.com/yoomimi/Image-denoiser/blob/main/src/cat_clean.jpg?raw=true) | ![Image 2](https://github.com/yoomimi/Image-denoiser/blob/main/src/cat_noisy.jpg?raw=true) | ![Image 3](https://github.com/yoomimi/Image-denoiser/blob/main/src/cat_result.jpg?raw=true) |
|:------------------------------------------:|:------------------------------------------:|:------------------------------------------:|
| Clean image                     | Noisy image                        | Denoised image                        |

<br>

__RMSE = 9.97 with snowman image (kernel size = 7)__

| ![Image 7](https://github.com/yoomimi/Image-denoiser/blob/main/src/snowman_clean.jpg?raw=true) | ![Image 8](https://github.com/yoomimi/Image-denoiser/blob/main/src/snowman_noisy.jpg?raw=true) | ![Image 9](https://github.com/yoomimi/Image-denoiser/blob/main/src/snowman_result.jpg?raw=true) |
|:------------------------------------------:|:------------------------------------------:|:------------------------------------------:|
| Clean image                     | Noisy image                        | Denoised image                        |

<br>

______



### [2] Bilateral filter

The bilateral filter is another non-linear filtering technique that reduces noise while preserving edges. It works by considering both the spatial closeness and the intensity difference of pixels. The bilateral filter computes a weighted average of nearby pixels, where the weights depend on both the spatial distance and the intensity difference.

1. For each pixel, define a neighborhood (or kernel).
2. Compute the spatial weight for each pixel in the kernel based on its distance from the center pixel.
3. Compute the intensity weight for each pixel based on its intensity difference from the center pixel.
4. Multiply the spatial and intensity weights and normalize them.
5. Compute the weighted sum of the pixel values in the neighborhood to get the new pixel value.


<br>

__RMSE = 10.39 with fox image (kernel size = 5, sigma_i = 50, sigma_s = 4)__

| ![Image 4](https://github.com/yoomimi/Image-denoiser/blob/main/src/fox_clean.jpg?raw=true) | ![Image 5](https://github.com/yoomimi/Image-denoiser/blob/main/src/fox_noisy.jpg?raw=true) | ![Image 6](https://github.com/yoomimi/Image-denoiser/blob/main/src/fox_result.jpg?raw=true) |
|:------------------------------------------:|:------------------------------------------:|:------------------------------------------:|
| Clean image                     | Noisy image                        | Denoised image                        |


______
