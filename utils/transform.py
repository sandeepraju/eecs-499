import cv2

def blur_image(img):
    # NOTE: 5x5 with std dev 0,0
    return cv2.GaussianBlur(img, (3,3), 0)
