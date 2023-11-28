import FaceDetection
import ImageLoader
import IntersectionOverUnion as IoU
import cv2
import numpy as np
from pprint import pprint


def evaluate():
    imgs = ImageLoader.gen_load_imgs("Data/wider_face_train_bbx_gt.txt")
    intersection_sum = 0
    union_sum = 0
    itr = 0

    for i in imgs:
        img = i.get("img", None)
        positions = i.get("positions", None)

        faces = FaceDetection.detect_faces(img)
        FaceDetection.draw_boxes(img, faces, (255, 0, 0))

        for position in positions:
            x_actual = position.get("x", None)
            y_actual = position.get("y", None)
            w_actual = position.get("width", None)
            h_actual = position.get("height", None)

            cv2.rectangle(
                img,
                (x_actual, y_actual),
                (x_actual + w_actual, y_actual + h_actual),
                (0,255,0)
            )

            #IoU
            for face in faces:
                x, y, w, h = face
                calculated_position = {"x": x, "y": y, "width": w, "height": h}
                if (IoU.rectangles_collide(position, calculated_position)):
                    if(position != None and calculated_position != None):
                        i_s, u_s = IoU.intersection_union(position, calculated_position)
                        if(i_s != None):
                            intersection_sum += i_s
                        if(u_s != None):
                            union_sum += u_s
        itr += 1
        if((itr % 20) == 0):
            intersection_over_union = intersection_sum / union_sum
            print("[No. of processed imgs:", itr, "]")
            print("[intersection over union percentage:] " , intersection_over_union)
        
        #remove the comment marks of the following 2 lines for getting a visualization
        #cv2.imshow("pictures", img)
        #cv2.waitKey(500)

evaluate()
print("Finished!")