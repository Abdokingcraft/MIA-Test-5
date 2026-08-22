# How does it detects

1- We turn the image into HSV

2- Splits the image into 2 channels (red and blue) using different mask for each color

3- We apply a kernel for each channel 

    - a 3x3 kernel for the red channel to remove noise around the shapes, this prevent it from detecting small points as circles
0 | 1 | 0
--- | --- | ---
1 | 1 | 1
0 | 1 | 0

![kernel_r](images/kernel_r.png)

    - a 5x5 kernel for the blue channel to fill up holes inside of shape so it would help out with the detection and reduce false positive

0 | 0 | 1 | 0 | 0
--- | --- | --- | --- | ---
0 | 1 | 1 | 1 | 0
1 | 1 | 1 | 1 | 1
0 | 1 | 1 | 1 | 0
0 | 0 | 1 | 0 | 0

![kernel_b](images/kernel_b.png)


### functions

**1- extract_ball** :
    it first measures all of the pixels on the filtered image so we can use it later
    ```
    total_pixels = cv2.countNonZero(mask)
    ```