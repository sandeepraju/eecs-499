import sys

# local modules
from utils import data

def main():
    output = sys.argv[1]
    
    # get all the files from input
    images = []
    for line in sys.stdin:
        images.append(line.strip())

    # convert the images to pickle
    data.img2pickle(images, output)
    
if __name__ == '__main__':
    main()
