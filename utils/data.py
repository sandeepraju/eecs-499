import cv2
import numpy as np

# define global constants
IN_IMG_WIDTH, IN_IMG_HEIGHT = 200, 200
OUT_IMG_WIDTH, OUT_IMG_HEIGHT = 150, 150
IMG_WIDTH, IMG_HEIGHT = 200, 200

from transform import blur_image

def img2hdf5(images, output):
    for i in xrange(len(images)):
        key = str(i).zfill(10)
        try:
            print 'processing:', images[i]
            img = load_image(images[i])
            
            # prepare x and y as needed
            x_img = prepare_image(img, (IMG_WIDTH, IMG_HEIGHT), (IN_IMG_WIDTH, IN_IMG_HEIGHT), blur_image)
            y_img = prepare_image(img, (IMG_WIDTH, IMG_HEIGHT), (OUT_IMG_WIDTH, OUT_IMG_HEIGHT))
        except Exception as e:
            print e
            continue

def img2pickle(images, output):
    # prepare empty container for x
    X = np.empty((1, 3, IN_IMG_WIDTH, IN_IMG_HEIGHT))
    Y = np.empty((1, 3, OUT_IMG_WIDTH, OUT_IMG_HEIGHT))

    for image in images:
        print 'processing:', image
        try:
            # load the image file
            img = load_image(image)

            # import ipdb; ipdb.set_trace()
            
            # prepare x and y as needed
            x_img = prepare_image(img, (IMG_WIDTH, IMG_HEIGHT), (IN_IMG_WIDTH, IN_IMG_HEIGHT), blur_image)
            y_img = prepare_image(img, (IMG_WIDTH, IMG_HEIGHT), (OUT_IMG_WIDTH, OUT_IMG_HEIGHT))
            
            # add to the dataset
            X = np.append(X, x_img[np.newaxis, :, :, :], 0)
            Y = np.append(Y, y_img[np.newaxis, :, :, :], 0)
        except Exception as e:
            print e
            continue

    # dump the dataset files
    # NOTE: skipping first entry since we initialized it to zero
    dump_dataset(X[1:,:,:,:], output + '_X.npy')
    dump_dataset(Y[1:,:,:,:], output + '_Y.npy')

def dump_dataset(dataset, file_path):
    np.save(file_path, dataset)

def load_dataset(file_path):
    return np.load(file_path)

def load_image(path):
     return cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32)
 
def prepare_image(img, original=(256, 256), resize=(256, 256), transform=None):
    h, w, _ = img.shape

    # ignore data files with inconsistencies
    if h != original[1] or w != original[0]:
        raise Exception("height and width don't match")

    # transform image if needed
    if transform:
        img = transform(img)

    # check if resize is needed
    if h != resize[1] or w != resize[0]: 
        img = cv2.resize(img, resize)


    # pre-processing based on the paper
    # x_img = x_img.astype(np.float32)
    # y_img = y_img.astype(np.float32)
    
    img[:,:,0] -= 103.939
    img[:,:,1] -= 116.779
    img[:,:,2] -= 123.68
    # img -= 127
    img *= 0.004
    
    # y_img[:,:,0] -= 103.939
    # y_img[:,:,1] -= 116.779
    # y_img[:,:,2] -= 123.68
    # # y_img -= 127
    # y_img *= 0.004

    # converting to keras format
    return np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)    


def generate_Y():
    # generate full white images
    # Y = np.full((500, 3, op_img_width, op_img_height), 255)
    # Y = np.concatenate((Y, np.full((435, 3, op_img_width, op_img_height), 0)), 0)
    pass


def ungenerate_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32)

    img[:,:,0] -= 103.939
    img[:,:,1] -= 116.779
    img[:,:,2] -= 123.68
    # img -= 127
    img *= 0.004
    
    return img

def generate_image(img, swap=True):
    img /= 0.004
    #out += 127
    img[:,:,0] += 103.939
    img[:,:,1] += 116.779
    img[:,:,2] += 123.68

    if swap:
        img = np.swapaxes(np.swapaxes(img[0,:,:,:], 0, 1), 1, 2)

    print 'saving prediction'
    # img = np.clip(img, 0, 255)
    # img /= 255.0
    # import ipdb; ipdb.set_trace()
    cv2.imwrite('/home/sandeep/tmp/network_output.png', img.astype(np.uint8))

