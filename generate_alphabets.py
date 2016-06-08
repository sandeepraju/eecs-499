import os
import sys
import random
from text import generate_white_patch, write_text, save_image

def lines_of_text(lines=10):
    text = []
    for i in xrange(lines):
        # shuffle the alphabets
        alphabets = "abcdefghijklmnopqrstuvwxyz"
        alpha_list = list(alphabets)
        random.shuffle(alpha_list)
        alpha_line = ''.join(alpha_list)

        # add spaces to make it look like words
        line = ""
        spaces = [1, 2, 3, 5, 7]
        while len(alpha_line) > 0:
            # pick a word len and generate that word
            word_len = random.choice(spaces)
            word = alpha_line[:word_len]
            alpha_line = alpha_line[word_len:]

            # should it be capitalized
            word = word if not random.choice([True, False]) else word.capitalize()
            line += word + " "

        text.append(line.strip())

    return text
    
def main():
    dst_dir = sys.argv[1]
    num_of_images = sys.argv[2]
    for i in xrange(1, int(num_of_images) + 1):
        text = lines_of_text(10)
        img = generate_white_patch((200, 200))
        img = write_text(img, text)

        filepath = os.path.join(dst_dir, 'img{0:05d}.png'.format(i))
        save_image(img, filepath)
        print 'saving', filepath

if __name__ == '__main__':
    main()
