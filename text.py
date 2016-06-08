import sys

import cv2
import numpy as np

def generate_white_patch(size):
    img = np.zeros((size[0], size[1], 3), np.uint8)
    img.fill(255)  # fill it with white

    return img

def write_text(img, text):
    # configure text
    text_color = (0, 0, 0)  # (R, G, B) tuple
    start_pos = (5, 20)
    font = cv2.FONT_HERSHEY_TRIPLEX  # https://codeyarns.com/2015/03/11/fonts-in-opencv/
    font_scale = 0.8
    font_height = 30 # for iteration
    thickness = 1
    num_chars_per_line = 12

    # LineType config
    # FILLED  = -1,
    # LINE_4  = 4, // 4-connected line
    # LINE_8  = 8, // 8-connected line
    # LINE_AA = 16 // antialiased line

    line_type = 16
    bottom_left_origin = False

    # place text on the image
    # handle multiple lines of text
    # REJECTED PULL REQUEST: https://github.com/Itseez/opencv/pull/313
    for line in text:
        cv2.putText(
            img, line, start_pos, font, font_scale, text_color,
            thickness, line_type, bottom_left_origin)
        start_pos = (start_pos[0], start_pos[1] + font_height)

    return img

def save_image(img, filepath):
    # save the image
    cv2.imwrite(filepath, img)


def main():
    filepath = sys.argv[1]

    # load the image
    # img = cv2.imread(filepath, cv2.IMREAD_COLOR).astype(np.float32)

    # generate image
    img = generate_white_patch((200, 200))
    
    # process the image
    text = """Duis dictum enim ut dolor venenatis, 
eget tristique felis tincidunt. Nulla finibus 
tellus sed nisi pellentesque, sed cursus risus commodo. 
Proin id elit et ipsum finibus posuere a nec nulla. 
Proin efficitur ligula vitae turpis tincidunt, nec 
placerat neque ornare. Etiam in commodo augue. 
Vivamus tristique massa quis justo malesuada blandit. 
Vestibulum turpis purus, aliquet ut risus quis, 
interdum lacinia risus. Ut mattis arcu in pharetra 
porttitor. Pellentesque rhoncus tincidunt dolor 
molestie sagittis. Cras ac elementum mauris. 
Maecenas tincidunt dui sit amet ipsum cursus finibus. 
Etiam et est massa. Duis molestie odio nisi, in 
semper sem congue at. Aliquam sit amet nulla tortor."""
    img = write_text(img, text.split('\n'))

    save_image(img, filepath)

if __name__ == '__main__':
    main()
