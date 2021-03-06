import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def main():
  # or wherever the src directory is
  sys.path.append('../../src')
  from openlif import lif as olif
  from openlif import find_homography, find_physical, find_interpolation

  # load camera calibration paramaters
  matrx = np.loadtxt('matrx.csv',delimiter=',')
  coeff = np.loadtxt('coeff.csv',delimiter=',')
  params = (matrx,coeff)

  # open images as video capture
  cap = cv.VideoCapture('still_water_images/still_water_%04d.tif', cv.CAP_IMAGES)

  # find water-air surface profile
  # in this case it is the upper most edge of the illuminated block
  # which is the 'low' interface in image coordinates
  kwargs = {
    'N':None, # defaults to None
    'calibration_params':params, # defaults to None
    'k':3, # defaults to 3
    'use_phase_mask':False, # defaults to True
    'connected':False, # defaults to True
    'use_column_max':False, # defaults to False
  }
  surface_air = olif(cap,'low','lower',**kwargs)

  # overlay surface line on original images
  try:
    os.mkdir('overlay_lif')
  except FileExistsError:
    pass

  cap.set(cv.CAP_PROP_POS_FRAMES, 0)
  for frame_number in range(surface_air.shape[0]):
    _, img = cap.read()
    img = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
    for i, j in enumerate(surface_air[frame_number,:]):
      img[int(j),i,:] = [0,0,2**16-1] #red

    cv.imwrite(f'overlay_lif/overlay_lif_{frame_number+1:04d}.png',img)

  # load calibration image and find homography
  calib = cv.imread('calib.tif',cv.IMREAD_GRAYSCALE)
  H, board_size = find_homography(calib,(9,6),25.4)

  # convert from pixel space to physical space
  xdata_air, ydata_air, R_air = find_physical(H,surface_air)
  xi, y, dx = find_interpolation( xdata_air,
                                 -ydata_air.filled(fill_value=np.nan),0.3)

  # generate plot for first frame (they are all identical)
  plt.plot(xi*dx,y[0,:],'k',linewidth=0.5)

  plt.ylabel('water surface elevation (mm)')
  plt.xlabel('position (mm)')
  plt.savefig('surface.png')
  
if __name__ == '__main__':
  main()
