import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

# or wherever the OpenLIF/src directory is
sys.path.append('../../src')
from openlif import lif as olif
from openlif import find_homography, find_physical, \
                    find_interpolation, find_adjustment, \
                    list_sequence_animation, \
                    list_adjust_data, \
                    list_merge_data

def main():
  # load camera calibration paramaters
  matrx = np.loadtxt('matrx.csv',delimiter=',')
  coeff = np.loadtxt('coeff.csv',delimiter=',')
  params = (matrx,coeff)

  # find water-air surface profile
  # in this case it is the upper most edge of the illuminated block
  # which is the 'low' interface in image coordinates
  kwargs = {
    'N':None, # defaults to None
    'calibration_params':params, # defaults to None
    'k':3, # defaults to 3
    'use_phase_mask':True, # defaults to True
    'connected':True, # defaults to True
    'use_column_max':False, # defaults to False
  }

  # fill a dictionary with data based on labels
  data = {}
  for loc in [ f'loc{n}' for n in range(2) ]:
    name = f'undular_bore_{loc}.avi'
    cap = cv.VideoCapture(name)
    surface_air = olif(cap,'low','lower',**kwargs)
    
    # load calibration image and find homography
    calib = cv.imread(f'calib_{loc}.tif', cv.IMREAD_GRAYSCALE)
    H, board_size = find_homography(calib,(9,6),25.4)
    
    # convert from pixel space to physical space
    xdata_air, ydata_air, R_air = find_physical(H,surface_air)
    xi, y, dx = find_interpolation( xdata_air,
                                   -ydata_air.filled(fill_value=np.nan),0.3)
    
    # in this case the auto-detection is upside down and backwards
    xi *= -1
    y  *= -1
    
    # zero the y axis
    y -= y[0,:]
    
    #ensure that xi is in increasing order
    idx_sort = np.argsort(xi)
    xi = xi[idx_sort]
    y  = y[:,idx_sort]
    
    # assign the physical location based on location number
    xi -= xi[0]
    if 'loc0' in loc:
      xi += 0
    if 'loc1' in loc:
      xi += int(260//dx)
    
    ti = np.arange(y.shape[0])
    dt = 1/125
    
    T,X = np.meshgrid(ti*dt,xi*dx,indexing='ij')
    plt.pcolormesh(X,T,y)
    plt.savefig(f'{loc}.png')
    plt.close()
    
    data[loc] = {}
    data[loc]['xi'] = xi
    data[loc]['dx'] = dx
    data[loc]['ti'] = ti
    data[loc]['dt'] = dt
    data[loc][ 'y'] = y

  # now we will aggregate collected data for sequential comparison
  locations = [ f'loc{n}' for n in range(2) ]
  tdata = [ data[loc]['ti'] for loc in locations ]
  xdata = [ data[loc]['xi'] for loc in locations ]
  ydata = [ data[loc][ 'y'] for loc in locations ]

  # find temporal and spatial offsets to match overlapping data
  adjust_t, adjust_x = list_adjust_data(tdata,xdata,ydata)

  for loc, adj_t in enumerate(adjust_t):
    print(f"adjust time for loc{loc+1} by {adj_t}")
    tdata[loc+1] += adj_t

  for loc, adj_x in enumerate(adjust_x):
    print(f"adjust space for loc{loc+1} by {adj_x}")
    xdata[loc+1] += adj_x

  # animate the adjusted data
  tdata = [ data[loc]['ti']/125 for loc in locations ]
  xdata = [ data[loc]['xi']*0.3 for loc in locations ]
  ydata = [ data[loc][ 'y'] for loc in locations ]

  xmin = sorted( [ np.nanmin(x) for x in xdata ] )[0 ] - 10
  xmax = sorted( [ np.nanmax(x) for x in xdata ] )[-1] + 10
  ymin = sorted( [ np.nanmin(y) for y in ydata ] )[0 ] - 1
  ymax = sorted( [ np.nanmax(y) for y in ydata ] )[-1] + 1

  anim = list_sequence_animation(xdata, ydata,
           name='aligned_anim', xlims=(xmin,xmax), ylims=(ymin,ymax))

  # merge data into a single data set
  tdata = [ data[loc]['ti'] for loc in locations ]
  xdata = [ data[loc]['xi'] for loc in locations ]
  ydata = [ data[loc][ 'y'] for loc in locations ]

  time, space, elevation = list_merge_data(tdata,xdata,ydata)

  # apply a low pass filter on the spatial axis
  window = signal.chebwin(151, at=100).reshape((1,-1))
  window = window/np.sum(window)
  elevation = signal.convolve(elevation,window,mode='valid')
  q,r = divmod(space.shape[0]-elevation.shape[1],2)
  space = space[q:-q+r]

  # animate merged and filtered data set
  xmin = space.min()*dx - 10
  xmax = space.max()*dx + 10
  ymin = elevation.min() - 1
  ymax = elevation.max() + 1

  anim = list_sequence_animation([space*dx], [elevation],
           name=f'merged_anim', xlims=(xmin,xmax), ylims=(ymin,ymax))

if __name__ == '__main__':
  main()
