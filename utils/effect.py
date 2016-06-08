def blur_image(img):
    # NOTE: 5x5 with std dev 0,0
    return cv2.GaussianBlur(img, (5,5), 0)
