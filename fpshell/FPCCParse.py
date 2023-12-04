#============for opencv approach========
import os

from .utils.FloorplanToBlenderLib import *

import numpy as np
import cv2
from PIL import Image

import json
from datetime import datetime

#============additional for ai approach======
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torch.nn.functional as F
# import cv2 
from torch.utils.data import DataLoader

from .model import get_model
from .utils.loaders import FloorplanSVG, DictToTensor, Compose, RotateNTurns
from .utils.plotting import segmentation_plot, polygons_to_image, draw_junction_from_dict, discrete_cmap
# discrete_cmap()
from .utils.post_prosessing import split_prediction, get_polygons, split_validation
from mpl_toolkits.axes_grid1 import AxesGrid
import shapely.geometry as sg

from matplotlib import cm




class RBGFloorPlanOpenCV():
    
    rbgjson = {
        "imagePath": "NameOfImage.png", 	#string, filename of the image
        "imageHeight": 0,			#integer, height in pixels
        "imageWidth": 0,
        "predictions": []
    }
    
    
    def writeJSON(jsondict, outputname):
        jsonobj = json.dumps(jsondict, indent=4)

        with open(outputname, "w") as jout:
            jout.write(jsonobj)
    
    
    def getOuterShell(img_path, timestamp, outputsFolder):
        # Read floorplan image
        img = cv2.imread(img_path)
        # img = Augmentations.autoResize(img_path)
        
        # Create blank image
        height, width, channels = img.shape
        blank_image = np.zeros((height,width,3), np.uint8)

        # Grayscale image
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        ogImgInfo = [
            img_path, #imgpath
            img.shape[0],    #y
            img.shape[1],    #x
        ]

        # detect outer Contours (simple floor or roof solution), paint them red on blank_image
        contourJSON, img = detect.detectOuterContours(detect_img=gray, imgInfo=ogImgInfo, output_img=blank_image, color=(255,0,0))

        outContourImgName = f"{outputsFolder}/{timestamp}-ContourOutput-{os.path.basename(img_path)}"

        # #cv2 write method
        # cv2.imwrite(outcontourImgName, img)
        contourPilImg = Image.fromarray(img)
        contourPilImg = contourPilImg.save(outContourImgName)
        #contour is already a formatted json
        
        return contourJSON

    def detectRooms(img_path, timestamp, outputsFolder):
        # img = cv2.imread(img_path)
        img = Augmentations.autoResize(img_path)
            # grayscale image
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # create verts (points 3d), points to use in mesh creations
        verts = []
        # create faces for each plane, describe order to create mesh points
        faces = []

        # Height of waLL
        height = 0.999

        # Scale pixel value to 3d pos
        scale = 100

        gray = detect.wall_filter(gray)

        gray = ~gray

        rooms, colored_rooms = detect.find_rooms(gray.copy())

        gray_rooms =  cv2.cvtColor(colored_rooms,cv2.COLOR_BGR2GRAY)

        # get box positions for rooms
        boxes, gray_rooms = detect.detectPreciseBoxes(gray_rooms, gray_rooms)

        # display(Image.fromarray(colored_rooms))
        outRoomImgName = f"{outputsFolder}/{timestamp}-RoomOutput-{os.path.basename(img_path)}"
        roomPilImg = Image.fromarray(colored_rooms)
        roomPilImg = roomPilImg.save(outRoomImgName)

        #Create verts
        room_count = 0
        for box in boxes:
            verts.extend([transform.scale_point_to_vector(box, scale, height)])
            room_count+= 1

        # create faces
        for room in verts:
            count = 0
            temp = ()
            for pos in room:
                temp = temp + (count,)
                count += 1
            faces.append([(temp)])

        vertOutputs = {
            "verts": verts
        }

        jsonName = f"{outputsFolder}/{timestamp}-RoomVerts-{os.path.basename(img_path).split('.')[0]}.json"
        RBGFloorPlanOpenCV.writeJSON(vertOutputs, jsonName)
    
    def detectWalls(img_path, timestamp, outputsFolder):
        # img = cv2.imread(img_path)
        img = Augmentations.autoResize(img_path)

        # grayscale image
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # create wall image (filter out small objects from image)
        wall_img = detect.wall_filter(gray)

        # detect walls
        boxes, img = detect.detectPreciseBoxes(wall_img)

        outWallImgName = f"{outputsFolder}/{timestamp}-WallOutput-{os.path.basename(img_path)}"
        wallPilImg = Image.fromarray(wall_img)
        wallPilImg = wallPilImg.save(outWallImgName)
        

        # create verts (points 3d), points to use in mesh creations
        verts = []
        # create faces for each plane, describe order to create mesh points
        faces = []

        # Height of waLL
        wall_height = 1

        # Scale pixel value to 3d pos
        scale = 100

        # Convert boxes to verts and faces
        verts, faces, wall_amount = transform.create_nx4_verts_and_faces(boxes, wall_height, scale)

        # Create top walls verts
        verts = []
        for box in boxes:
            verts.extend([transform.scale_point_to_vector(box, scale, 0)])

        # create faces
        faces = []
        for room in verts:
            count = 0
            temp = ()
            for _ in room:
                temp = temp + (count,)
                count += 1
            faces.append([(temp)])

        vertOutputs = {
            "verts": verts
        }

        jsonName = f"{outputsFolder}/{timestamp}-WallVerts-{os.path.basename(img_path).split('.')[0]}.json"
        RBGFloorPlanOpenCV.writeJSON(vertOutputs, jsonName)



class RBGFloorPlanPyTorch():
    def runCubicasaSegmentation(img_path, outPath, fulljson, imageOnly=False):
        rot = RotateNTurns()
        room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room" ,"Bed Room", "Bath",
                        "Entry", "Railing", "Storage", "Garage", "Undefined"]
        icon_classes = ["No Icon", "Window", "Door", "Closet", "Electrical Applience" ,"Toilet", "Sink",
                        "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]

        model = get_model('hg_furukawa_original', 51)
        n_classes = 44
        split = [21, 12, 11]
        model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
        model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
        checkpoint = torch.load('fpshell/model/cubicasa5k_model_best_val_loss_var.pkl', map_location=torch.device('cpu'))
        
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        # model.cuda()

        # Create tensor for pytorch
        # img = cv2.imread(img_path)
        img = Augmentations.autoResize(img_path)
        print(f"img dims: {img.shape}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # correct color channels

        # Image transformation to range (-1,1)
        img = 2 * (img / 255.0) - 1

        # Move from (h,w,3)--->(3,h,w) as model input dimension is defined like this
        img = np.moveaxis(img, -1, 0)

        # Convert to pytorch, enable cuda
        # img = torch.tensor([img.astype(np.float32)]).cuda()
        img = torch.tensor([img.astype(np.float32)]).cpu()
        n_rooms = 12
        n_icons = 11

        with torch.no_grad():
            #Check if shape of image is odd or even
            size_check = np.array([img.shape[2],img.shape[3]])%2

            #need to cast these as int, otherwise type is numpy.intc
            height = int(img.shape[2] - size_check[0]) 
            width = int(img.shape[3] - size_check[1])
            print(f"--------------height/width check: {height}({type(height)} x {width}({type(width)}) -----------")
            
            img_size = (height, width)

            rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
            pred_count = len(rotations)
            prediction = torch.zeros([pred_count, n_classes, height, width])
            for i, r in enumerate(rotations):
                forward, back = r
                # We rotate first the image
                rot_image = rot(img, 'tensor', forward)
                pred = model(rot_image)
                # We rotate prediction back
                pred = rot(pred, 'tensor', back)
                # We fix heatmaps
                pred = rot(pred, 'points', back)            #torch.Tensor
                # print(f"-----------pred: {pred}({type(pred)})")
                # We make sure the size is correct
                pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)       #throws TypeError: expected size to be one of int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], but got size with types [<class 'numpy.intc'>, <class 'numpy.intc'>]
                # We add the prediction to output
                prediction[i] = pred[0]

        prediction = torch.mean(prediction, 0, True)

        rooms_pred = F.softmax(prediction[0, 21:21+12], 0).cpu().data.numpy()
        rooms_pred = np.argmax(rooms_pred, axis=0)

        icons_pred = F.softmax(prediction[0, 21+12:], 0).cpu().data.numpy()
        icons_pred = np.argmax(icons_pred, axis=0)

        heatmaps, rooms, icons = split_prediction(prediction, img_size, split)
        polygons, types, room_polygons, room_types = get_polygons((heatmaps, rooms, icons), 0.2, [1, 2])
        
        #============add all types of polygons=================
        # print(f"-------------------------types: {types}-----------------------")
        wall_polygon_numbers=[i for i,j in enumerate(types) if j['type']=='wall']
        # print(f"-----------room_classes: {len(room_classes)}")
        # print(f"-----------icon_classes: {len(icon_classes)}")
        # print(f"-----------polygons: {len(polygons)}")
        # print(f"-----------room_types: {len(room_types)}")
        
        #icon is a class that includes the windows and door blues 
        boxes=[]
        #add the walls
        # polygonsParsed = []
        for i,j in enumerate(polygons):
            if i:
            # if i in wall_polygon_numbers:
                ttype = "rooms" if types[i]['type'] == "room" else types[i]['type']
                newPrediction = {
                        # "confidence": 0.95,
                        "confidence":  types[i]["prob"] if "prob" in types[i].keys() else 1.0,
                        'class': ttype,
                        # "name": f'{str(i)}-{types[i]["class"]}',
                        "name": icon_classes[types[i]["class"]],
                        'points': []
                }
                temp=[]
                # tempParsed = []
                for k in j:
                    temp.append(np.array([k]))
                    # tempParsed.append([int(k[0]), int(k[1])])
                    newPrediction["points"].append({"x":int(k[0]), "y":int(k[1])})
                boxes.append(np.array(temp))
                # polygonsParsed.append(tempParsed)
                fulljson["predictions"]+=[newPrediction]
                
        # Height of waLL
        wall_height = 1
        # Scale pixel value to 3d pos
        scale = 100
        verts, faces, wall_amount = transform.create_nx4_verts_and_faces(boxes, wall_height, scale)

        # Create top walls verts
        verts = []
        for box in boxes:
            verts.extend([transform.scale_point_to_vector(box, scale, 0)])

        # create faces
        faces = []
        for room in verts:
            count = 0
            temp = ()
            for _ in room:
                temp = temp + (count,)
                count += 1
            faces.append([(temp)])

        allpolygons = [] 
        # print(f"------------{room_types}-------------")
        for i,t in enumerate(room_types):
            # print(f"{t['type']},{t['class']}: {str(room_polygons[i])}: {dir(room_polygons[i])}")
            if not room_polygons[i].is_empty:
                ttype = "rooms" if t['type'] == "room" else t['type']
                tclass = room_classes[t['class'] - 1]
                # print(f"{t['type']},{t['class']}:{type(room_polygons[i])}")
                if type(room_polygons[i]) == type(sg.MultiPolygon()):
                    # print(f"{len(room_polygons[i].geoms)}")
                    print(f"in multipolygon: {room_polygons[i].geoms}")
                    for g in room_polygons[i].geoms:
                        newPrediction = {
                            "confidence": 0.95,
                            'class': ttype,
                            "name": tclass,
                            'points': []
                        }
                        # print(f"{g}: {list(g.exterior.coords)}: {type(g.exterior.coords)}")
                        points = [{"x":int(p[0]), "y":int(p[1])} for p in list(g.exterior.coords)]
                        newPrediction["points"] += points
                        allpolygons.append(newPrediction)
                elif type(room_polygons[i]) == type(sg.MultiLineString()):
                    print(f"in multilinestring: {room_polygons[i].geoms}")
                    newPrediction = {
                        "confidence": 0.95,
                        'class': ttype,
                        "name": tclass,
                        'points': []
                    }
                    points = [{"x":int(p[0]), "y":int(p[1])} for p in list([lig.coords for lig in room_polygons[i].geoms])]
                    newPrediction["points"]+=points
                    allpolygons.append(newPrediction)
                    
                else:
                    print(f"else case: {type(room_polygons[i])}")
                    newPrediction = {
                        "confidence": 0.95,
                        'class': ttype,
                        "name": tclass,
                        'points': []
                    }
                    
                    # print(f"room_polygons[i]: {room_polygons[i]} | {type(room_polygons[i])}")
                    # print(f"room_polygons[i]: {room_polygons[i].boundary}")
                    # print(f"{list(room_polygons[i].boundary.coords)}")
                    # print(list(room_polygons[i].exterior.coords))
                    # points = [{"x":int(p[0]), "y":int(p[1])} for p in list(g.exterior.coords)]
                    
                    #need to return to this later
                    print(f"room_polygons[i] options: {dir(room_polygons[i])}")
                    print(f"interiors: {room_polygons[i].interiors}")
                    try:
                        points = [{"x":int(p[0]), "y":int(p[1])} for p in list(room_polygons[i].boundary.coords)]   #does not work for all Polygon instances
                        # points = [{"x":int(p[0]), "y":int(p[1])} for p in list(room_polygons[i].coords)]
                    except Exception as e:
                        print(e)
                        points = [{"x":0, "y":0}]
                    
                    newPrediction["points"]+=points
                    allpolygons.append(newPrediction)

        fulljson["predictions"]+=allpolygons
        return fulljson
    
        # jsonobj = json.dumps(fulljson, indent=4)
        # jsonname = f"jsons/{os.path.basename(img_path).split('.')[0]}.json"
        # with open(jsonname, "w") as jfile:
        #     jfile.write(jsonobj) 
        
        # #==========logic for saving predictions as images============
        
        # pol_room_seg, pol_icon_seg = polygons_to_image(polygons, types, room_polygons, room_types, height, width)

        # if imageOnly:
        #     #--------for purposes of creating TF2-like synthetic data, combining icons and rooms
        #     print(f"pol_room_seg: {pol_room_seg}")
        #     print(f"pol_icon_seg: {pol_icon_seg}")
        #     #------use PIL to fill in pixels
        #     roomIm = Image.fromarray(np.uint8(cm.gist_earth(pol_room_seg)*255))
        #     outname = os.path.basename(img_path).split(".")[0]+"-PIL"+"."+os.path.basename(img_path).split(".")[1]
        #     outpath = os.path.join(outPath, outname)
        #     print(f"------outname: {outname}")
        #     roomIm.save(outname)
            
        #     # #---attempt using plot lib figure
        #     # fig = plt.figure(figsize=(12,12))
        #     # w = int(img.shape[0])
        #     # h = int(img.shape[1])
        #     # fig.set_size_inches(w,h)
        #     # ax = plt.Axes(fig, [0., 0., 1., 1.])
        #     # ax.set_axis_off()
        #     # fig.add_axes(ax)
        #     # ax.imshow(fig, aspect='auto')
        #     # dpi=96
        #     # outname = os.path.basename(img_path).split(".")[0]+"-SOLVE"+os.path.basename(img_path).split(".")[1]
        #     # outpath = os.path.join(outPath, outname)
        #     # fig.savefig(outpath, dpi)
        # else:
        #     plt.figure(figsize=(12,12))
        #     ax = plt.subplot(1, 1, 1)
        #     ax.axis('off')
        #     # rseg = ax.imshow(pol_room_seg, cmap='rooms', vmin=0, vmax=n_rooms-0.1)    #ValueError: 'rooms' is not a valid value for cmap; 
        #     rseg = ax.imshow(pol_room_seg, vmin=0, vmax=n_rooms-0.1)
        #     cbar = plt.colorbar(rseg, ticks=np.arange(n_rooms) + 0.5, fraction=0.046, pad=0.01)
        #     cbar.ax.set_yticklabels(room_classes, fontsize=20)
        #     plt.tight_layout()
            
        #     outname = os.path.basename(img_path).split(".")[0]+"-SOLVE"+os.path.basename(img_path).split(".")[1]
        #     outpath = os.path.join(outPath, outname)
        #     plt.savefig(outpath)        #main solve
        #     # plt.show()
            
        #     plt.figure(figsize=(12,12))
        #     ax = plt.subplot(1, 1, 1)
        #     ax.axis('off')
        #     iseg = ax.imshow(pol_icon_seg, cmap='jet', vmin=0, vmax=n_icons-0.1)
        #     cbar = plt.colorbar(iseg, ticks=np.arange(n_icons) + 0.5, fraction=0.046, pad=0.01)
        #     cbar.ax.set_yticklabels(icon_classes, fontsize=20)
        #     plt.tight_layout()
            
        #     outname = os.path.basename(img_path).split(".")[0]+"-SOLVE-icons"
        #     outpath = os.path.join(outPath, outname)
        #     plt.savefig(outpath)
        #     # plt.show()


#==============augmentation functions===========
class Augmentations():
    #larger images exponentially increase eval time
    def autoResize(imgpath):
        #maxH = 3000
        maxW = 2000     #width tends to be the largest dimension, so images will tend to scale on this basis
        img = cv2.imread(imgpath)
        if img.shape[1] > maxW:
            #then img width 
            scale_ratio = maxW/img.shape[1]
            newH = int(img.shape[0]*scale_ratio)
            newW = int(img.shape[1]*scale_ratio)
            newDim = (newW,newH)
            resized = cv2.resize(img, newDim, interpolation=cv2.INTER_AREA)
            print(f"resized {imgpath} to {newW}x{newH}")
            return resized
        else:
            return img


class CCInference():
    def makeInference(savepath, outFolder=None):
        # timenow = datetime.now().strftime("%d%m%Y%S%M%H")
        timenow = "0"
        
        #this module is outside the standard django structure, so needs the full app path
        if outFolder is not None:
            outFolder = f"fpshell/{outFolder}"
        
        t1 = datetime.now()
        #views should provide the imgfullpath, named as savepath
        shellJSON = RBGFloorPlanOpenCV.getOuterShell(savepath, timenow, outFolder)      #aka contourjson
        finalJSON = RBGFloorPlanPyTorch.runCubicasaSegmentation(savepath,outFolder, fulljson=shellJSON, imageOnly=True)   
        
        t2 = datetime.now()
        t3 = t2-t1
        print(f"Completed solve in {t3}, saved in {outFolder}") 

        return finalJSON
    