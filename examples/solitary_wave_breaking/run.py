import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#sys.path.append('../../src') # or wherever the src directory is
from openlif import lif as olif
from openlif import find_homography, find_physical, find_interpolation

def main():
  # load camera calibration paramaters
  matrx = np.loadtxt('matrx.csv',delimiter=',')
  coeff = np.loadtxt('coeff.csv',delimiter=',')
  params = (matrx,coeff)

  # open images as video capture
  cap = cv.VideoCapture('solitary_wave_breaking.avi')

  # find water-air surface profile
  # in this case it is the upper most edge of the illuminated block
  # which is the 'low' interface in image coordinates
  kwargs = {
    'N':None, # defaults to None
    'calibration_params':params, # defaults to None
    'k':4, # defaults to 3 - need more sensitivity for these images
    'use_phase_mask':True, # defaults to True
    'connected':True, # defaults to True
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

  # calibration detection is backwards from what we want - reverse axis
  xi *= -1
  y  *= -1

  # generate colormesh plot
  dt = 1/125
  ti = np.arange(y.shape[0])
  T, X = np.meshgrid(ti,xi,indexing='ij')

  cmap = plt.pcolormesh(X*dx,T*dt,y)

  plt.xlabel('position (mm)')
  plt.ylabel('time (s)')
  plt.colorbar(cmap,label='water surface elevation (mm)')
  plt.tight_layout()
  plt.savefig('surface.png',dpi=400)


if __name__ == '__main__':
  main()
