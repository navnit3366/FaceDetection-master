import re
import cv2
from pprint import pprint
import numpy as np


def read_file(path_to_file):
    imgname = ""
    positions = []
    metadata_list = []
    position_pattern = re.compile("\d+ \d+ \d+ \d+ [012] [01] [01] [01] [012] [01]")

    with open(path_to_file) as f:
        for line in f:
            if line.endswith("jpg\n"):
                if(imgname != "" and len(positions) != 0):
                    metadata = {"imgname" : imgname , "positions" : positions}
                    metadata_list.append(metadata)
                    positions = []
                imgname = line.strip()
            
            elif (position_pattern.match(line)):
                 values = line.split(" ")
                 dimension = {
                    "x" : int(values[0]) , 
                    "y" : int(values[1]) , 
                    "width" : int(values[2]) , 
                    "height" : int(values[3])
                 }
                 positions.append(dimension)
        
        if(imgname != "" and len(positions) != 0):
            metadata = {"imgname" : imgname , "positions" : positions}
            positions = []
            metadata_list.append(metadata)
            imgname = line.strip()    
        
    return metadata_list


def gen_load_imgs(path_to_file):
    metadata_list = read_file(path_to_file)
    print("[Total image count:]", len(metadata_list))
    
    for metadata in metadata_list:
        path = metadata.get("imgname", None)
        
        if (path != None):
            path = "Data/" + path
            #print(path)
        
        img = cv2.imread(path, 1)
        img_with_metadata = {"img" : img , "positions" : metadata.get("positions")}
        
        yield img_with_metadata

def load_resized_imgs(path_to_file, size):
    imgs = gen_load_imgs(path_to_file)

    for img in imgs:
        image = img.get("img", None)

        height = np.size(image, 0)
        width = np.size(image, 1)
        scalefactor_width = size / width
        scalefactor_height = size / height

        image = cv2.resize(image, (size,size))

        scaled_img = {"img" : image, "positions": img.get("positions"), "sfw" : scalefactor_width, "sfh" : scalefactor_height }
        yield scaled_img

def visualize_resized_imgs(path_to_file, size):
    # imgs = load_resized_imgs(path_to_file, size)
    imgs = load_cropped_imgs(path_to_file, size)

    for img in imgs:
        positions = img.get("positions", [])
        sfw = img.get("sfw", 1)
        sfh = img.get("sfh", 1)

        if (positions != None):
            for p in positions:
                x = p.get("x") * sfw
                y = p.get("y") * sfh
                w = p.get("width") * sfw
                h = p.get("height") * sfh

                cv2.rectangle(
                    img.get("img"),
                    (int(x), int(y)),
                    (int(x+w), int(y+h)),
                    (0,255,0)
                )    

        print(img.get("img").shape)
        print(type(img.get("img")))
        cv2.imshow("resized image", img.get("img"))
        cv2.waitKey(1000)


def load_cropped_imgs(path_to_file, size):
    """
    Problem: nicht alle geladenen Bilder haben den shape 600x600x3
    Lösung: es gibt Bilder, deren width = 0 ist. Diese kackvögel...
    """
    imgs = gen_load_imgs(path_to_file)

    for img in imgs:

        image = img.get("img", None)
        positions = img.get("positions", None)
        new_positions = []

        width = np.size(image, 1)
        height = np.size(image, 0)

        if(width >=600 and height >= 600):
            pos_x = int(width/2) - int(size/2) - 1
            pos_y = int(height/2) - int(size/2) - 1

            image = image[pos_y:(pos_y+size), pos_x:(pos_x+size)]
            # print(type(image))

            for position in positions:
                x = position.get("x", None)
                y = position.get("y", None)
                w = position.get("width", None)
                h = position.get("height", None)


                if((x >= pos_x) and (x <= size) and ((x+w) <= size)):
                    if((y >= pos_y) and (y <= size) and ((y+h) <= size)):
                        x = x - pos_x
                        y = y - pos_y
                        position = {"x": x, "y": y, "width": w, "height": h}
                        new_positions.append(position)
            
            if(image.shape == (600,600,3)):
                new_image = {"img": image, "positions": new_positions, "sfw": 1, "sfh": 1}
                yield new_image