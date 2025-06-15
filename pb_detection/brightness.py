from openpyxl import Workbook  
import numpy as np
import os
from skimage import io
import cv2

def calculate_brightness(image_path,npy_path): 
                  
  image = cv2.imread(image_path,cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
  dat = np.load(npy_path, allow_pickle=True).item()
  masks = dat['masks']
  roi_number = np.max(masks)
  if roi_number != 0:
    [x,y] = image.shape
    pixel_number = 0
    
    pbody_brightness = 0
    bg_brightness = 0
    pixel_number_bg = 0
  
    for i in range(0,x):
      for j in range(0,y):
        if ( masks[i][j]!= 0):
          pixel_number = pixel_number+1
        if ( masks[i][j]== 0):
          pixel_number_bg = pixel_number_bg+1 
    for i2 in range(0,x):
      for j2 in range(0,y):
        if ( masks[i2][j2]!= 0):
          pbody_brightness = pbody_brightness+image[i2][j2]/pixel_number 
        if ( masks[i2][j2]== 0):
          bg_brightness = bg_brightness+image[i2][j2]/pixel_number_bg
    relative_brightness = pbody_brightness-bg_brightness
  else:
    pbody_brightness = 'empty'
    bg_brightness = 'empty'
    relative_brightness = 'empty'
  return pbody_brightness, bg_brightness, pixel_number, roi_number
  
def pbody_brightness(image_path,pbody_position,background_brightness): 
  image = cv2.imread(image_path,cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
  n= len(pbody_position)
  region_total_brightness = np.zeros([n,1])
  for i in range(0,n):
        [x1, y1, x2, y2] = np.array(pbody_position[i].cpu()).astype('int8')
        region = image[y1:y2, x1:x2]
        region_size = (x2 - x1) * (y2 - y1)
        region_total_brightness[i] = np.mean(region.astype(np.int64))- background_brightness
        print(region_total_brightness[i])
  
  aver_brightness = np.mean(region_total_brightness)
  var_brightness = np.var(region_total_brightness)
  
  return aver_brightness,var_brightness
