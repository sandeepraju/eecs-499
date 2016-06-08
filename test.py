import sys

from utils import data

def main():
    filename = sys.argv[1]
    print 'un-generating image'
    img = data.ungenerate_image(filename)

    print 'generating image'
    data.generate_image(img, swap=False)

if __name__ == '__main__':
    main()
