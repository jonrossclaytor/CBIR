from PIL import Image
import imagehash

def hashImage(path):
    '''Function that accepts a full path to an image and retuns a hash for that image as a string'''
    imghash = imagehash.average_hash(Image.open(path))
    return str(imghash)