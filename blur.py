import cv2
import numpy as np

from utils import transform

def main():
    img = cv2.imread('/home/sandeep/tmp/img01.png', cv2.IMREAD_COLOR).astype(np.float32)
    img = transform.blur_image(img)
    cv2.imwrite('/home/sandeep/tmp/img01-blur.png', img)

if __name__ == '__main__':
    main()
