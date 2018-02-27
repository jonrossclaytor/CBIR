import dicom
import urllib
from bs4 import BeautifulSoup
import os
import matplotlib.pyplot as plt

def downloadImages(only_dicom=False,
                    images_loc='http://rasinsrv04.cstcis.cti.depaul.edu/RSNA/all_tf/',
                    folder_name = 'images_download'):

    # connect to the website
    conn = urllib.urlopen(images_loc)
    url_html = conn.read()
    soup = BeautifulSoup(url_html, 'html.parser')

    # make a folder to store images
    os.mkdir(folder_name)
    os.chdir('./'+folder_name)

    # iterate through the images in the directory and save the right files

    for a_tag in soup.find_all('a', href=True): # iterate through all the links
        image_id = a_tag['href'] # this is unique for each image
        image_loc = images_loc + '/' + image_id # image is a bs4 element obj

        
        # TODO: current saving to disk and then reading is tedious, try a simpler way
        urllib.urlretrieve(image_loc, 'temp') # save the file as temp
        
        try:
            temp = dicom.read_file('temp') # to check it dicom... files have no extensions, so not sure of better way
            os.rename('temp',image_id)
            print 'dicom: '+image_id
        except:
            try:
                temp = plt.imread('temp') # check if file is an image
                os.rename('temp',image_id)
                print 'non-dicom: ', image_id
            except:
                print 'broke: ', image_loc