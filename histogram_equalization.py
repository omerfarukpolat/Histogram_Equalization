import numpy as np
import cv2 as cv

img_filename = 'test2.jpg'
save_filename = 'output_1.jpg'
save_filename2 = 'output_2.jpg'


def q1():
    img = cv.imread(img_filename, 0)

    # aldığım image'ı arraye çevir
    img_array = np.array(img)

    histogram_array = [0] * 256

    for i in img_array.flatten():
        histogram_array[i] = histogram_array[i] + 1


    # num_pixels = N*M
    num_pixels = np.sum(histogram_array)

    # arraydeki her değeri N*M (num_pixels) değerine böl
    histogram_array = histogram_array / num_pixels

    # cdf al
    cdf_histogram_array = np.cumsum(histogram_array)


    # 8 bitle gösterilecek en büyük sayı = 255 ile her değeri çarp ve round et
    # böylece transform edilmiş olacak

    transformed_map = np.round_(255 * cdf_histogram_array).astype(np.uint8)

    # transform tablosuna bakarak orjinal resimdeki her değeri transform map'teki
    # round edilmiş değer ile değiştir
    img_list = list(img_array.flatten())

    for i in range(0, len(img_list)):
        img_list[i] = transformed_map[img_list[i]]


    eq_image = [[0 for x in range(img_array[0].size)] for y in range(img_array[0].size)]
    for i in range(0, img_array[0].size):
        for j in range(0, img_array[0].size):
            eq_image[i][j] = img_list[(i * img_array[0].size) + j]


    # çıkan arrayi resime dönüştür ve kaydet
    eq_image_array = np.array(eq_image)

    img_opencv = cv.imread('test1.jpg',0)
    equ_opencv = cv.equalizeHist(img_opencv)

    res = np.hstack((img_opencv, equ_opencv))

    abs_array = np.sum(abs(eq_image - equ_opencv))

    cv.imshow('q1: my hist. equ.', eq_image_array)
    cv.imshow('q1: opencv hist. equ.', res)
    cv.imshow('difference:{}'.format(abs_array), abs(eq_image - equ_opencv)*255)
    cv.waitKey(0)
    cv.destroyAllWindows()

    print("difference between our histogram equalization and opencv's histogram equalization: ", abs_array)


def q2():
    img = cv.imread('test1.jpg', 0)
    h_img_array = [0] * 256

    min_val = 11111
    min_index = -1
    img_array = np.array(img)
    img_list = list(img_array.flatten())


    for i in range(0, len(h_img_array)):
        if h_img_array[i] < min_val:
            min_val = h_img_array[i]
            min_index = i

    for i in img_array.flatten():
        h_img_array[i] = (h_img_array[i] + 1)


    c_image_array = [0] * len(h_img_array)
    c_image_array[0] = h_img_array[0]

    for i in range(1, len(h_img_array)):
        c_image_array[i] = c_image_array[i - 1] + h_img_array[i]

    c_image_array = np.array(c_image_array)
    h_min = c_image_array[min_index]

    for item in c_image_array:
        if item != 0 and item < h_min:
            h_min = item
    transformed_himage = [0] * len(c_image_array)

    for i in range(0, len(c_image_array)):
        transformed_himage[i] = np.round(((c_image_array[i] - h_min) / (len(img_array.flatten()) - h_min)) * (len(h_img_array) - 1)).astype(
            np.uint8)

    for i in range(0, len(img_list)):
        img_list[i] = transformed_himage[img_list[i]]

    eq_image_2 = [[0 for x in range(img_array[0].size)] for y in range(img_array[0].size)]
    for i in range(0, img_array[0].size):
        for j in range(0, img_array[0].size):
            eq_image_2[i][j] = img_list[(i * img_array[0].size) + j]

    equ_opencv = opencvHistogramEqualization()
    eq_image_array2 = np.array(eq_image_2)
    abs_array = np.sum(eq_image_array2 - equ_opencv)

    print("q2: difference between our histogram equalizations: ", abs_array)
    cv.imshow('output_3', eq_image_array2)
    cv.imshow('opencv', equ_opencv)
    cv.imshow('difference:{}'.format(abs_array),equ_opencv - eq_image_array2)
    cv.waitKey(0)
    cv.destroyAllWindows()

def opencvHistogramEqualization():
    img_opencv = cv.imread('test1.jpg', 0)
    equ_opencv = cv.equalizeHist(img_opencv)

    res = np.hstack((img_opencv, equ_opencv))
    return equ_opencv

if __name__ == '__main__':
    q1()
    q2()