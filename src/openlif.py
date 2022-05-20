import os
import math

import cv2 as cv
import scipy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import describe, linregress
from scipy.signal import detrend
from matplotlib.animation import FuncAnimation

#~~~~~~~~~~~~~~~HELPER FUNCTIONS FOR IDENTIFYING SURFACE LINE~~~~~~~~~~~~~~~~~~
# these functions help identify the surface line in PLIF images

def _get_frame(cap: cv.VideoCapture, N: int) -> np.ndarray :
  """
  Get the Nth frame from the video capture in grayscale

  Return the nth frame from an opencv video capture object as greyscale or
  None if it fails.

  Raises TypeError for some inputs. Raises IndexError if N is out of bounds.
  Raises AssertionError is video capture is not open.
  """
  if not isinstance(cap,cv.VideoCapture):
    raise TypeError("cap must be an opencv video capture object")
  elif not cap.isOpened():
    raise AssertionError("cap must be open")
  elif not isinstance(N,int):
    raise TypeError("N must be an int")

  frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
  # Apparently, frameCount == -2147483648 or -1 for single image sequence
  if frame_count < 0:
    frame_count = 1

  if not 0<=N<frame_count:
    raise IndexError("N must be positive and <= frame count of cap")

  # cap.set is expensive, only use if needed
  if cap.get(cv.CAP_PROP_POS_FRAMES) != N:
    cap.set(cv.CAP_PROP_POS_FRAMES, N)

  ret_frame, frame = cap.read()
  
  if ret_frame:
    if len(frame.shape) == 2:
      pass # already greyscale
    elif frame.shape[2] == 3:
      frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    elif frame.shape[2] == 4:
      frame = cv.cvtColor(frame, cv.COLOR_BGRA2GRAY)
    else:
      raise TypeError("video source not supported")
    
    return frame
  
  else:
    return None


def _get_grad_phase(src: np.ndarray) -> "tuple of np.ndarray" :
  """
  Return the gradient and phase of the grayscale image

  Return the gradient and phase of a grayscale image or None if it fails.
  Uses Scharr gradient estimation. Normalizes quantites to use the entire
  dynamic range of the src image data type.

  Raises TypeError for some inputs.
  """
  if not isinstance(src,np.ndarray):
    raise TypeError("src must be a numpy array")
  if not (src.dtype == np.uint8  or src.dtype == np.uint16):
    raise TypeError("src must have type np.uint8 or np.uint16")

  gradx = cv.Scharr(src, cv.CV_32F, 1, 0, 3)
  grady = cv.Scharr(src, cv.CV_32F, 0, 1, 3)

  grad  = cv.magnitude(gradx, grady)
  phase = cv.phase(gradx, grady)

  if src.dtype == np.uint8:
    kwargs = {'alpha':0,'beta':255,'norm_type':cv.NORM_MINMAX,
              'dtype':cv.CV_8UC1}
  else: # otherwise np.uint16
    kwargs = {'alpha':0,'beta':65535,'norm_type':cv.NORM_MINMAX,
              'dtype':cv.CV_16UC1}

  grad  = cv.normalize(grad , grad , **kwargs)
  phase = cv.normalize(phase, phase, **kwargs)

  return grad, phase


def _get_mask_from_gradient(src: np.ndarray, k: int) -> np.ndarray :
  """
  Identifies large values of an image gradient with a binary mask.

  Return a binary mask isolating the values of src that are sufficiently
  large. Sufficiently large is determined by clustering the image in to k
  parts, then defining the background as the cluster with the largest number
  of elements. All other clusters are considered sufficently large and their
  locations in the image are marked 1 in the binary mask. The background
  is marked 0 in the binary mask.

  Raises TypeError for some inputs.
  """
  if not isinstance(src,np.ndarray):
    raise TypeError("src must be a numpy array")
  if not (src.dtype == np.uint8  or src.dtype == np.uint16):
    raise TypeError("src must have type np.uint8 or np.uint16")

  # Prepare the src for clustering
  clusterable = np.array(src.ravel(), dtype=np.float32)

  # kmeans requires some initial guess to iteratively improve
  # Using this inital label seems to be more reliable than using PP or random
  labels = np.zeros(clusterable.shape, dtype=np.int32)
  labels[ np.argwhere(clusterable == clusterable.max()) ] = k-1
  
  # generate and shape label array
  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 20, 1.0)

  _, labels, centers = cv.kmeans(clusterable, k, labels, criteria, 1,
                                                cv.KMEANS_USE_INITIAL_LABELS)
  
  labels = labels.reshape(src.shape[0],src.shape[1])
  # exclude the background label from a binary mask where the background label
  # has the smallest gradient value among the cluster centers, all other labels
  # are included. The background label can be identified by noting that the
  # center values are organized like: center[label] = gradient_value
  dst = np.ones(src.shape, dtype=src.dtype)
  dst[ labels == np.argmin(centers) ] = 0
  
  return dst


def _get_mask_from_phase(src: np.ndarray, mask: np.ndarray,
                         direction: "'low' or 'high'") -> np.ndarray :
  """
  Identifies the low or high phase of an image gradient with a binary mask.

  Return a binary mask identifying a low valued cluster or the high valued
  cluster as indicated by the directio input. The background cluster is
  assumed to be the cluster with the largest count and is ignored.

  Raises a TypeError or a ValueError for some inputs.
  """
  if not isinstance(src,np.ndarray):
    raise TypeError("src must be a numpy array")
  elif not isinstance(mask,np.ndarray):
    raise TypeError("mask must be a numpy array")
  elif not (src.dtype == np.uint8  or src.dtype == np.uint16):
    raise TypeError("src must have type np.uint8 or np.uint16")
  elif not (mask.dtype == np.uint8  or mask.dtype == np.uint16):
    raise TypeError("mask must have type np.uint8 or np.uint16")
  elif not len(src.shape) == len(mask.shape) == 2:
    raise ValueError("src and mask must have two dimensions (grayscale)")
  elif not (direction == 'low' or direction == 'high'):
    raise ValueError("direction must be 'low' or 'high'")

  # make them the same dtype but preserve the dynamic range of src
  if src.dtype != mask.dtype:
    mask = np.array(mask,dtype=mask.dtype)

  # identify the foreground cluster with the correct directionality
  clusterable = np.array(np.multiply(src,mask).ravel(), dtype=np.float32)
  labels = np.zeros(clusterable.shape,dtype=np.int32)
  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 20, 1.0)
  
  # phase is normalized to take up the entire dynamic range, so choose to
  # split the mask down the middle into an 'low' and 'high' phase
  mid = 255//2 if (src.dtype == np.uint8) else 65535//2
  # low phase is in the lower half and nonzero
  labels[ np.argwhere(np.logical_and(clusterable > 0, clusterable < mid)) ] = 1
  # high phase is in the upper half
  labels[ np.argwhere(clusterable > mid) ] = 2
  
  # TODO: determine if this clustering actually improves results
  # compared to a simple binary threshold
  _, labels, centers = cv.kmeans(clusterable, 3, labels, criteria, 1,
                                    cv.KMEANS_USE_INITIAL_LABELS  )
  labels = np.array(labels.reshape(src.shape[0],src.shape[1]), dtype=src.dtype)
  
  # To identify the low and high labels, must also identify the background
  # label which is assumed to be the largest group by count
  
  # recall phase data is clustered like: centers[label] = phase_val
  label_by_count = np.argsort(np.bincount(labels.ravel()))
  label_by_phase = np.argsort(centers.ravel())
  background_label = label_by_count[-1]
  
  label_by_phase_excluding_background = np.delete(
    label_by_phase, np.where(label_by_phase == background_label))
  
  low_label  = label_by_phase_excluding_background[ 0]
  high_label = label_by_phase_excluding_background[-1]
  
  choose_label = int(low_label) if direction=='low' else int(high_label)
  
  return cv.compare(labels,(choose_label,0,0,0),cv.CMP_EQ)


def _get_widest_connected_group(mask: np.ndarray) -> np.ndarray:
  '''
  Identifes the widest group (uppermost in case of ties) in the binary image.

  Find the widest connected group in the binary mask. If there are multiple,
  choose the uppermost among them. Requires an uint8 type image but assumes
  that the input image is a binary mask (no check).

  Raises a TypeError for some inputs.
  '''
  if not isinstance(mask,np.ndarray):
    raise TypeError("mask must be a numpy array")
  elif not (mask.dtype == np.uint8):
    raise TypeError("mask must have type np.uint8")
  
  num_groups, labels, stats, centroids = \
                      cv.connectedComponentsWithStats(mask,connectivity=8)
  
  # identify candidates of connected components by area
  idx_candidates = np.argsort(stats[:,cv.CC_STAT_AREA])[:-1]
  # among the valid candidates, sort by width of connected components
  stats_width = stats[idx_candidates,cv.CC_STAT_WIDTH]
  widest_groups = np.argwhere(stats_width == np.amax(stats_width))
  # among the widest groups, choose the one closes to top of image
  # recall that the y axis for images is flipped
  top_group = np.argmin(stats[idx_candidates,cv.CC_STAT_TOP][widest_groups])
  
  # create a new mask from the label of the widest & highest cluster
  mask_new = np.zeros(labels.shape, dtype=bool)
  label = idx_candidates[widest_groups[top_group]]
  mask_new[labels == label] = 1
  
  return np.multiply(mask,mask_new)


def _get_mask_maxima(grad: np.ndarray, mask: np.ndarray) -> np.ndarray:
  """
  Finds the local maxima of an image gradeint where the mask is 1.

  Returns a binary mask where the values are local maxima or a plateau
  edge of grad. Applies the input mask before finding the local maxima.
  Assumes (no check) that the mask is binary.

  Raises a TypeError for some inputs.
  """
  if not isinstance(grad,np.ndarray):
    raise TypeError("grad must be a numpy array")
  elif not isinstance(mask,np.ndarray):
    raise TypeError("mask must be a numpy array")
  elif not (mask.dtype == np.uint8  or mask.dtype == np.uint16):
    raise TypeError("mask must have type np.uint8 or np.uint16")
  
  se = np.array([1,0,1],dtype=np.uint8).reshape(-1,1)
  grad_masked = np.multiply(grad,mask)
  local_max = cv.dilate(grad_masked, se)
  local_max = cv.compare(grad_masked,local_max,cv.CMP_GE)
  return np.multiply(local_max,mask)


def _get_surfaceline(mask: np.ndarray, side: "'lower' or 'upper'") \
                                                                -> np.ndarray:
  """
  Identifes the surface line from a binary mask.

  Returns a 1 dimensional numpy array with the pixel values of the uppermost
  or lowermost values in mask.

  Raises a TypeError or ValueError for some inputs.
  """
  if not isinstance(mask,np.ndarray):
      raise TypeError("mask must be a numpy array")
  elif not (mask.dtype == np.uint8  or mask.dtype == np.uint16):
      raise TypeError("mask must have type np.uint8 or np.uint16")
  elif not (side=='upper' or side=='lower'):
      raise ValueError("direction must be 'low' or 'high'")
  # TODO: why convert uint8 or uint16 into binary mask?
  # just require a binary array in the first place?
  # accept any non-zero value of the mask, mask must be converted to binary
  mask = mask>0
  
  n,m = mask.shape
  if side=='upper':
    args = (0,n,n)
  else: # side=='lower'
    args = (n,0,n)
  
  weight_y = np.linspace(*args,dtype=int).reshape(-1,1).repeat(m,axis=1)

  line = np.argmax(weight_y*mask,axis=0)
  
  # TODO: replace this with numpy functions
  # when columns are all 0, line returns an invalid point, replace with -1
  for i, j in enumerate(line):
    if mask[j,i]==0:
        line[i] = -1
  
  return line.ravel()


def _get_supersample(line: np.ndarray, grad: np.ndarray) -> np.ndarray:
  """
  Identifes the supersample interpolation along the surface line of grad.

  Returns a tuple of 1 dimensional numpy arrays. The first returns line
  with values replaced to be negative if the supersample is invalid. The second
  returns the supersample of the gradient or 0 if the supersample is invalid.

  Negative values in the first array correspond to the following meanings:
    -1 : no identified maxima in column
    -2 : identified maxima is not a local maxima (all equal)
    -3 : identified maxima is not a local maxima (on a line)

  Raises a TypeError or ValueError for some inputs.
  """
  if not isinstance(line,np.ndarray):
      raise TypeError("line must be a numpy array")
  elif not isinstance(grad,np.ndarray):
      raise TypeError("grad must be a numpy array")
  elif not len(line.shape) == 1:
      raise ValueError("line must have one dimension")
  elif not len(grad.shape) == 2:
      raise ValueError("grad must have two dimensions")
  
  supersample = np.zeros(line.shape)
  
  # TODO: replace loop with array operations
  for i,j in enumerate(line):
    try:
      upper  = int(grad[j-1,i])
      center = int(grad[j  ,i])
      lower  = int(grad[j+1,i])
    except IndexError:
      line[i] = -1
      continue
    
    numerator = upper - lower
    denominator = 2*upper + 2*lower - 4*center
    
    if j == -1:
      pass
    elif upper==center and lower==center and upper==lower:
      line[i] = -2
    elif numerator!=0 and denominator==0:
      line[i] = -3
    else:
      supersample[i] = numerator/denominator
    
    # useful for debugging
    #if not np.isfinite(supersample).all():
    #  print(f"non-finite value at {i}, {j}")
    #  print(f"numerator: {numerator}")
    #  print(f"denominator: {denominator}")
    #  raise ValueError

  return line, supersample


# The following functions each handle different combinations of the input
# values to lif(), this is explicit but perhaps too verbose.

def _loop_phase_mask_connected(cap: cv.VideoCapture, num_frames: int, k: int,
                               direction: "'low' or 'high'",
                               side: "'lower' or 'upper'") -> np.ndarray:
  '''
  Performs LIF for a specific case in the lif function.

  Assumes valid input.
  Considers the the case:
    use_column_max = False
    use_phase_mask = True
    connected = True
    calibration_params = None
  '''
  width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
  surface = np.empty((num_frames,width))
  
  for num in range(num_frames):
    frame = _get_frame(cap,num)
    
    grad, phase = _get_grad_phase(frame)
    mask = _get_mask_from_gradient(grad, k)
    
    mask_phase = _get_mask_from_phase(phase,mask,direction)
    mask_connected = _get_widest_connected_group(mask_phase)
    mask_maxima = _get_mask_maxima(grad,mask)*mask_connected
    
    line = _get_surfaceline(mask_maxima,side)
    line , supersample = _get_supersample(line, grad)
    surface[num,:] = line + supersample
  
  return surface


def _loop_phase_mask_connected_calibrate(cap: cv.VideoCapture,
                                          num_frames: int, k: int,
                                          direction: "'low' or 'high'",
                                          side: "'lower' or 'upper'",
                                          calibration_params: tuple,) \
                                          -> np.ndarray:
  '''
  Performs LIF for a specific case in the lif function.

  Assumes valid input.
  Considers the the case:
    use_column_max = False
    use_phase_mask = True
    connected = True
    calibration_params = Tuple
  '''
  width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
  surface = np.empty((num_frames,width))
  
  for num in range(num_frames):
    frame = _get_frame(cap,num)
    frame = cv.undistort(frame, calibration_params[0], calibration_params[1])
    
    grad, phase = _get_grad_phase(frame)
    mask = _get_mask_from_gradient(grad, k)
    
    mask_phase = _get_mask_from_phase(phase,mask,direction)
    mask_connected = _get_widest_connected_group(mask_phase)
    mask_maxima = _get_mask_maxima(grad,mask)*mask_connected
    
    line = _get_surfaceline(mask_maxima,side)
    line , supersample = _get_supersample(line, grad)
    surface[num,:] = line + supersample
  
  return surface


def _loop_phase_mask(cap: cv.VideoCapture, num_frames: int, k: int,
                     direction: "'low' or 'high'",
                     side: "'lower' or 'upper'") -> np.ndarray:
  '''
  Performs LIF for a specific case in the lif function.

  Assumes valid input.
  Considers the the case:
    use_column_max = False
    use_phase_mask = True
    connected = False
    calibration_params = None
  '''
  width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
  surface = np.empty((num_frames,width))
  
  for num in range(num_frames):
    frame = _get_frame(cap,num)
    
    grad, phase = _get_grad_phase(frame)
    mask = _get_mask_from_gradient(grad, k)
    
    mask_phase = _get_mask_from_phase(phase,mask,direction)
    mask_maxima = _get_mask_maxima(grad,mask)*mask_phase
    
    line = _get_surfaceline(mask_maxima,side)
    line , supersample = _get_supersample(line, grad)
    surface[num,:] = line + supersample
  
  return surface


def _loop_phase_mask_calibrate(cap: cv.VideoCapture, num_frames: int, k: int,
                                direction: "'low' or 'high'",
                                side: "'lower' or 'upper'",
                                calibration_params: tuple) -> np.ndarray:
  '''
  Performs LIF for a specific case in the lif function.

  Assumes valid input.
  Considers the the case:
    use_column_max = False
    use_phase_mask = True
    connected = False
    calibration_params = None
  '''
  width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
  surface = np.empty((num_frames,width))
  
  for num in range(num_frames):
    frame = _get_frame(cap,num)
    frame = cv.undistort(frame, calibration_params[0], calibration_params[1])
    
    grad, phase = _get_grad_phase(frame)
    mask = _get_mask_from_gradient(grad, k)
    
    mask_phase = _get_mask_from_phase(phase,mask,direction)
    mask_maxima = _get_mask_maxima(grad,mask)*mask_phase
    
    line = _get_surfaceline(mask_maxima,side)
    line , supersample = _get_supersample(line, grad)
    surface[num,:] = line + supersample
  
  return surface


def _loop_local_maxima(cap: cv.VideoCapture, num_frames: int, k: int,
                       direction: "'low' or 'high'",
                       side: "'lower' or 'upper'") -> np.ndarray:
  '''
  Performs LIF for a specific case in the lif function.

  Assumes valid input.
  Considers the the case:
    use_column_max = False
    use_phase_mask = False
    connected = False
    calibration_params = None
  '''
  width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
  surface = np.empty((num_frames,width))
  
  for num in range(num_frames):
    frame = _get_frame(cap,num)
    
    grad, phase = _get_grad_phase(frame)
    mask = _get_mask_from_gradient(grad, k)
    
    mask_maxima = _get_mask_maxima(grad,mask)
    
    line = _get_surfaceline(mask_maxima,side)
    line , supersample = _get_supersample(line, grad)
    surface[num,:] = line + supersample
  
  return surface


def _loop_local_maxima_calibrate(cap: cv.VideoCapture, num_frames: int,
                                  k: int, direction: "'low' or 'high'",
                                  side: "'lower' or 'upper'",
                                  calibration_params: tuple) -> np.ndarray:
  '''
  Performs LIF for a specific case in the lif function.

  Assumes valid input.
  Considers the the case:
    use_column_max = False
    use_phase_mask = False
    connected = False
    calibration_params = None
  '''
  width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
  surface = np.empty((num_frames,width))
  
  for num in range(num_frames):
    frame = _get_frame(cap,num)
    frame = cv.undistort(frame, calibration_params[0], calibration_params[1])
    
    grad, phase = _get_grad_phase(frame)
    mask = _get_mask_from_gradient(grad, k)
    
    mask_maxima = _get_mask_maxima(grad,mask)
    
    line = _get_surfaceline(mask_maxima,side)
    line , supersample = _get_supersample(line, grad)
    surface[num,:] = line + supersample
  
  return surface


def _loop_maxima(cap: cv.VideoCapture, num_frames: int, k: int,
                 direction: "'low' or 'high'",
                 side: "'lower' or 'upper'") -> np.ndarray:
  '''
  Performs LIF for a specific case in the lif function.

  Assumes valid input.
  Considers the the case:
    use_column_max = True
    use_phase_mask = False
    connected = False
    calibration_params = None
  '''
  width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
  surface = np.empty((num_frames,width))
  
  for num in range(num_frames):
    frame = _get_frame(cap,num)
    
    grad, _ = _get_grad_phase(frame)
    
    mask_maxima = np.zeros(grad.shape, dtype=np.uint8)
    mask_maxima[np.argmax(grad,axis=0),np.arange(width)] = 1
    
    line = _get_surfaceline(mask_maxima,side)
    line , supersample = _get_supersample(line, grad)
    surface[num,:] = line + supersample
  
  return surface


def _loop_maxima_calibrate(cap: cv.VideoCapture, num_frames: int, k: int,
                            direction: "'low' or 'high'",
                            side: "'lower' or 'upper'",
                            calibration_params: tuple) -> np.ndarray:
  '''
  Performs LIF for a specific case in the lif function.

  Assumes valid input.
  Considers the the case:
    use_column_max = True
    use_phase_mask = False
    connected = False
    calibration_params = tuple
  '''
  width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
  surface = np.empty((num_frames,width))
  
  for num in range(num_frames):
    frame = _get_frame(cap,num)
    frame = cv.undistort(frame, calibration_params[0], calibration_params[1])
    
    grad, _ = _get_grad_phase(frame)
    
    mask_maxima = np.zeros(grad.shape, dtype=np.uint8)
    mask_maxima[np.argmax(grad,axis=0),np.arange(width)] = 1
    
    line = _get_surfaceline(mask_maxima,side)
    line , supersample = _get_supersample(line, grad)
    surface[num,:] = line + supersample
  
  return surface


def lif(cap: cv.VideoCapture, direction: "'low' or 'high'",
         side: "'lower' or 'upper'", N: "int or None" = None,
         calibration_params: "tuple or None" = None, k: int = 3,
         use_phase_mask : bool = True, connected : bool = True,
         use_column_max : bool = False) -> np.ma.array:
  '''
  Performs lif analysis on an opencv video capture.

  Imports each frame from cap as a grayscale image and performs LIF analysis
  on each frame. Returns identified elevation of the surface line as a
  numpy array with shape (N,M) where N is as specified or the number of frames
  in cap (if unspecified) and M is the width of the images in cap.

  The argument 'direction' refers to the direction of the gradient where 'low'
  roughly corresponds with pi radians, and 'high' roughly corresponds to 3 pi
  radians. The argument 'side' refers to which side of masked regions it will
  attempt to identify, where 'lower' is the lowermost index value, and 'upper'
  is the uppermost index value within the mask. The argument 'k' allows for
  adjusting the sensitivity when identifying large gradients, higher values of
  k mean more compute time but allows for smaller local gradient maxima. The
  argument calibration_params should be a tuple with two values where the
  first value in the tuple is the camera matrix and the second value is the
  distortion coefficients as in OpenCV's undistort. The argument use_phase_mask
  is a boolean to specify if the phase of the gradient should be used to
  identify the surface. The argument connected is a boolean to specify if the
  identify surface should be connected (will only return a connected surface).
  The agrument use_column_max is used to determine if a global maximum should be
  used to identify surface. If use_column_max is True then use_phase_mask and
  connected arguments are ignored.

  Raises a TypeError or ValueError for some inputs.
  '''
  if not isinstance(cap,cv.VideoCapture):
    raise TypeError("cap must be an opencv video capture object")
  elif not (direction == 'low' or direction == 'high'):
    raise ValueError("direction must be 'low' or 'high'")
  elif not (side == 'lower' or side == 'upper'):
    raise ValueError("side must be 'lower' or 'upper'")
  elif not (isinstance(N,int) or N is None):
    raise ValueError("N must be an int or None")
  elif not (isinstance(k,int) and k>1):
    raise ValueError("k must be an int greater than 1")
  elif not isinstance(use_phase_mask,bool):
    raise ValueError("use_phase_mask must be a bool")
  elif not (isinstance(calibration_params,tuple) \
                                          or calibration_params is None):
    raise TypeError("calibration_params must be tuple or None")
  elif not ( calibration_params is None or (type(calibration_params) is tuple
                                          and len(calibration_params) == 2)):
    raise ValueError("calibration_params must be tuple with two values")
  elif not isinstance(use_column_max,bool):
    raise ValueError("use_column_max must be a bool")
  
  num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) if N is None else N
  
  if calibration_params is None:
    args = (cap,num_frames,k,direction,side)
    
    if   use_column_max:
      surface = _loop_maxima(*args)
    elif use_phase_mask and connected:
      surface = _loop_phase_mask_connected(*args)
    elif use_phase_mask and not connected:
      surface = _loop_phase_mask(*args)
    else:
      surface = _loop_local_maxima(*args)
  
  else:
    args = (cap,num_frames,k,direction,side,calibration_params)
    
    if   use_column_max:
      surface = _loop_maxima_calibrate(*args)
    elif use_phase_mask and connected:
      surface = _loop_phase_mask_connected_calibrate(*args)
    elif use_phase_mask and not connected:
      surface = _loop_phase_mask_calibrate(*args)
    else:
      surface = _loop_local_maxima_calibrate(*args)
  
  return np.ma.masked_less(surface,0)

#~~~~~~~~~~~~~~~~~~HELPER FUNCTIONS FOR PLOTTING~~~~~~~~~~~~~~~~~~

def list_sequence_animation(xdata: list, ydata: list, name: str ='anim',
      fig: "None or matplotlib figure" = None,
      ax: "None or matplotlib axis" = None,
      xlims: "None or tuple" = None,
      ylims: "None or tuple" = None ) -> "matplotlib FuncAnimation" :
  """
  Write an animation of the provided data.

  Writes out an H264 encoded animation of the data by default. Each data
  in the lists is animated with a different color, so that overlapping
  measurements may be inspected manually.
  """
  if not isinstance(xdata, list):
    raise TypeError("xdata must be a list")
  elif not isinstance(ydata, list):
    raise TypeError("ydata must be a list")
  elif not isinstance(name, str):
    raise TypeError("name must be a string")
  elif not (fig is None or isinstance(fig, matplotlib.figure.Figure)):
    raise TypeError("fig must be a matplotlib figure")
  elif not (ax is None or isinstance(ax, matplotlib.axes.Axes)):
    raise TypeError("ax must be a matplotlib axis")
  elif not (xlims is None or isinstance(xlims,tuple)):
    raise TypeError("xlims must be None or tuple")
  elif not (ylims is None or isinstance(ylims,tuple)):
    raise TypeError("ylims must be None or tuple")
  elif isinstance(xlims,tuple) and not len(xlims)==2:
    raise ValueError("xlims must have length 2")
  elif isinstance(ylims,tuple) and not len(ylims)==2:
    raise ValueError("ylims must have length 2")

  prop_cycle = plt.rcParams['axes.prop_cycle']
  colors = prop_cycle.by_key()['color']

  if fig is None and ax is None:
    fig, ax = plt.subplots()
  elif fig is not None and ax is not None:
    pass
  else:
    return None

  if xlims is not None:
    ax.set_xlim(xlims)
  if ylims is not None:
    ax.set_ylim(ylims)

  lines = []
  for i in range(len(xdata)):
    lobj = ax.plot([], [], lw=2, color=colors[i])[0]
    lines.append(lobj)

  def init():
    for line in lines:
        line.set_data([],[])
    return lines

  def animate(t):
    for lnum,line in enumerate(lines):
        line.set_data(xdata[lnum], ydata[lnum][t,:])

    return line,

  num_frames = sorted([y.shape[0]-1 for y in ydata])[0]

  anim = FuncAnimation(fig, animate, init_func=init,
                       frames=num_frames, interval=20, blit=True)

  anim.save(name+'.mp4', fps=30, writer='ffmpeg')

  return anim

#~~~~~~~~~~~~~~~~~~HELPER FUNCTIONS FOR FINDING HOMOGRAPHY~~~~~~~~~~~~~~~~~~
# These functions are designed to help process calibration board images into a
# homography matrix.

def _find_chessboard_points(img: np.ndarray, board_size: tuple,
                            write_dir: "string or None" = None) -> np.ndarray :
  """
  Identify points on a chessboard image

  Identifies the chessboard point in a greyscale image, returning None if it
  is not able to find one of the specified size. Will write a sequence of
  images with the identified chessboard points to write_dir if a chessboard
  is found and write_dir is specified.

  Raises a TypeError or ValueError for some inputs.
  """
  if not isinstance(img,np.ndarray):
    raise TypeError("img must be a numpy array")
  elif not (len(img.shape)==2):
    raise ValueError("img must have two dimensions")
  elif not isinstance(board_size,tuple):
    raise TypeError("board_size must be a tuple")
  elif not (len(board_size)==2):
    raise ValueError("board_size must have two items")
  elif not (isinstance(write_dir,str) or write_dir is None):
    raise TypeError("write_dir must be a str or None")

  if isinstance(write_dir,str):
    if not os.path.isdir(write_dir):
      raise ValueError("write_dir must be a valid directory")

  flag, corners = cv.findChessboardCorners(img,board_size)
  if flag:
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
    image_points = cv.cornerSubPix(img,corners,(11,11),(-1,-1),criteria)
    if write_dir is not None: # in this case it must be a valid directory
      print_chessboard_corners(img,image_points,board_size,point_dir)
  
  elif not flag:
    return None
  
  return image_points


def _create_worldpoints_grid(board_size: tuple, square_size: 'int or float') \
                                                                -> np.ndarray:
  """ makes world points for the specified grid """
  if not (len(board_size)==2):
    raise ValueError("board_size must have two items")
  elif not isinstance(board_size[0],(int,float)):
    raise TypeError("board_size[0] must be an int or float")
  elif not isinstance(board_size[1],(int,float)):
    raise TypeError("board_size[1] must be an int or float")
  elif not isinstance(square_size,(int,float)):
    raise TypeError("square_size must be an int or float")
  
  x = np.arange(0,board_size[0],1,np.float32) * square_size
  y = np.arange(0,board_size[1],1,np.float32) * square_size
  X, Y = np.meshgrid(x,y)
  return np.stack((X.ravel(),Y.ravel()),axis=1)


def find_homography(img: np.ndarray, board_size: tuple,
                    square_size: 'positive int or float',
                    ret_points: bool = False) -> tuple :
  """
  Attempts to find a homogaphy from a calibration board image.

  Finds a homography from a calibration board with board size equal to
  or less than the provided size, and greater than or equal to (3,3)

  Raises a TypeError or ValueError for some inputs. Raises an
  AssertionError if no checkerboard is found in the image.
  """
  if not isinstance(img,np.ndarray):
    raise TypeError("img must be a numpy array")
  elif not (len(img.shape) == 2):
    raise ValueError("img must have two dimensions")
  elif not isinstance(board_size,tuple):
    raise TypeError("board_size must be a tuple")
  elif not (len(board_size) == 2):
    raise ValueError("board_size must have two items")
  elif not isinstance(square_size,(int,float)):
    raise TypeError("square_size but be an int or float")
  elif not (square_size > 0):
    raise ValueError("square_size non-zero and positive")
  
  # generate a list of possible grid sizes
  sizes = []
  rng = range(board_size[1],3-1,-1)
  for width in range(board_size[0],3-1,-1):
    sizes.append(zip((width,)*len(rng),rng))
  
  sizes = [item for subzip in sizes for item in subzip]
  
  # increment through sizes until a valid board is found
  counter, image_points = 0, None
  while image_points is None and counter < len(sizes):
    board_size = sizes[counter]
    image_points = _find_chessboard_points(img,board_size)
    counter += 1
  
  # if a board is not found, raise an error
  assert image_points is not None, "unable to find a checkerboard in image"
  
  world_points = _create_worldpoints_grid(board_size,square_size)
  H, _ = cv.findHomography(image_points, world_points)
  
  if ret_points:
    return H, board_size, image_points, world_points
  
  return H, board_size

#~~~~~~~~~~~~~~~~~~HELPER FUNCTIONS FOR PIXEL TO PHYSICAL~~~~~~~~~~~~~~~~~~
# These functions are designed to help convert pixel location data into
# physical location data.

def _find_lineartrend(xdata: np.ma.MaskedArray, ydata: np.ma.MaskedArray) \
                                                              -> np.ndarray :
  """
  Identify a linear trend in the data.

  Identify the slope of the linear trend for the given xdata and ydata where
  outliers are removed. xdata and ydata must be one dimensional arrays. Inliers
  are determined by lying 3 standard deviations out after detrending. The
  return matrix, R, is a rotation matrix with rotation taken about the z axis,
  or the optical axis in the case of pixel data.
  """
  if not isinstance(xdata,np.ma.MaskedArray):
    raise TypeError("xdata must be a numpy masked array")
  elif not (len(xdata.shape)==1):
    raise ValueError("xdata must have one dimensions")
  elif not isinstance(ydata,np.ma.MaskedArray):
    raise TypeError("ydata must be a numpy masked array")
  elif not (len(ydata.shape)==1):
    raise ValueError("ydata must have one dimensions")
  elif not (xdata.shape==ydata.shape):
    raise ValueError("xdata and ydata must have the same shape")
  
  data = np.ma.column_stack((xdata,ydata))
  valid_data = np.ma.compress_rows(data)
  
  y_detrend = detrend(valid_data[:,1])
  _, _, mean, var, _, _ = describe(y_detrend)
  std = math.sqrt(var)
  
  valid_data[:,1] = np.ma.masked_where(np.abs(y_detrend - mean) > 4*std,
                                                              valid_data[:,1])
  valid_data = np.ma.compress_rows(valid_data)
  
  slope = linregress(valid_data[:,0],valid_data[:,1])[0]
  theta = -np.arctan(slope)
  
  # construct a rotation matrix from the angle
  R = np.array([
      [np.cos(theta),-np.sin(theta),0],
      [np.sin(theta), np.cos(theta),0],
      [0            , 0            ,1]
  ])
  
  return R


def _apply_homography(H: np.ndarray, vdata: np.ndarray) -> tuple :
  """
  Apply a homography, H, to pixel data where only v of (u,v,1) is needed.

  Apply a homography to pixel data where only v of the (u,v,1) vector
  is given. It is assumed that the u coordinate begins at 0.
  The resulting vector (x,y,z) is normalized by z to find (x,y,1)
  """
  if not isinstance(H,np.ndarray):
    raise TypeError("H must be a numpy array")
  elif not (H.shape==(3,3)):
    raise ValueError("H must have shape (3,3)")
  elif not isinstance(vdata,np.ma.MaskedArray):
    raise TypeError("vdata must be a numpy masked array")
  elif not (len(vdata.shape)==2):
    raise ValueError("vdata must have two dimensions")
  
  # build stack of (u,v,1) vectors
  N,M = vdata.shape
  u, v = np.arange(0,M,1), np.arange(0,N,1)
  udata = np.ma.array(np.meshgrid(u,v)[0] ,mask=vdata.mask)
  wdata = np.ma.array(np.ones(vdata.shape),mask=vdata.mask)
  data = np.ma.stack((udata.ravel(),vdata.ravel(),wdata.ravel()),axis=-1).T
  
  # apply H but ignore columns which have any masked values
  valid_data = np.matmul(H,np.ma.compress_cols(data))
  # normalize by the second index
  for i in range(3):
    valid_data[i,:] = np.divide(valid_data[i,:],valid_data[2,:])
  
  # extract valid values into array with original shape
  idx = np.ma.array(np.arange(data.shape[1]),mask=vdata.ravel().mask)
  valid_idx = np.ma.compressed(idx)
  data = np.zeros((2,data.shape[1]))
  data[0,valid_idx] = valid_data[0,:]
  data[1,valid_idx] = valid_data[1,:]
  data = data.reshape(2,N,M)
  
  return np.ma.array(data[0,:,:],mask=vdata.mask), \
         np.ma.array(data[1,:,:],mask=vdata.mask)


def _is_rotationmatrix(R: np.ndarray, tol: float = 1e-6) -> bool:
  """ returns True if R is a rotation matrix and False otherwise """
  if not isinstance(R,np.ndarray):
    raise TypeError("R must be a numpy array")
  elif not (isinstance(tol,float) and tol > 0):
    raise TypeError("tol must be a positive float")
  
  if not (len(R.shape)==2 and R.shape[0]==R.shape[1]):
    return False
  
  Rt = np.transpose(R)
  Rt_dot_R = np.dot(Rt, R)
  I = np.identity(R.shape[0], dtype = R.dtype)
  n = np.linalg.norm(I - Rt_dot_R)
  
  return n < tol


def _apply_rotationmatrix(R: "rotation matrix", xdata: np.ndarray,
                                                ydata: np.ndarray) -> tuple :
  """ applies the rotation matrix R to the vector (x,y,1) for each item in
  xdata and ydata """
  if not isinstance(R,np.ndarray):
    raise TypeError("R must be a numpy array")
  elif not (R.shape==(3,3)):
    raise ValueError("R must have shape (3,3)")
  elif not _is_rotationmatrix(R):
    raise ValueError("R must be a rotation matrix")
  elif not isinstance(xdata,np.ma.MaskedArray):
    raise TypeError("xdata must be a numpy masked array")
  elif not isinstance(ydata,np.ma.MaskedArray):
    raise TypeError("ydata must be a numpy masked array")
  elif not (xdata.shape==ydata.shape):
    raise ValueError("xdata and ydata must have the same shape")
  
  N,M = ydata.shape
  mask = ydata.mask
  zdata = np.ma.ones((N,M))
  data = np.matmul(R,np.stack((xdata.data,ydata.data,zdata.data),axis=0)
                                                 .reshape(3,-1)).reshape(3,N,M)
  
  return np.ma.array(data[0,:,:],mask=mask), \
         np.ma.array(data[1,:,:],mask=mask)

def find_physical(H: np.ndarray, vdata: np.ma.MaskedArray,
                  R: 'np.ndarray or None' = None, zero: bool = True) -> tuple :
  """
  Finds the physical values associated with the surface line of lif data.
  
  Apply a homography, H, to pixel data then remove linear trends either by
  utilizing the specified value of R, or finding a trend when R is None.
  The first index, assumed to be the spatial index, is forced to increase in
  increase in value with indices with a 180 degree (pi radian) rotation about
  the z axis if needed. If the pixel data is taken with positive downward, as
  is typical, the y axis will point downward, and the z axis will point 'into
  the paper.' The median value of the y data is set to zero by default, as
  indicated with zero=True.

  Returns TypeError or ValueError for some inputs.
  """
  if not isinstance(H,np.ndarray):
    raise TypeError("H must be a numpy array")
  elif not (H.shape==(3,3)):
    raise ValueError("H must have shape (3,3)")
  elif not isinstance(vdata,np.ma.MaskedArray):
    raise TypeError("vdata must be a numpy masked array")
  elif not (len(vdata.shape)==2 and vdata.shape[0]>1):
    raise ValueError("vdata must have two dimensions and dimension 0 must \
                                                        be greater than 1")
  elif not (isinstance(R,np.ndarray) or R is None):
    raise TypeError("R must be numpy ndarray or None")
  elif not isinstance(zero,bool):
    raise TypeError("zero must be bool")
  
  xdata, ydata = _apply_homography(H,vdata)
  
  if R is None:
    R = _find_lineartrend(xdata[0,:],ydata[0,:])
  
  xdata, ydata = _apply_rotationmatrix(R,xdata,ydata)
  
  idx_sort = sorted([xdata.argmin(),xdata.argmax()])
  xleft  = xdata.ravel()[idx_sort[0]]
  xright = xdata.ravel()[idx_sort[1]]
  
  if zero:
    isfine = np.isfinite(ydata[~ydata.mask])
    ydata = ydata - np.ma.median(ydata[~ydata.mask][isfine])
  
  return xdata, ydata, R

#~~~~~~~~~~~~~~~~~~HELPER FUNCTIONS FOR CONFORMING DATA~~~~~~~~~~~~~~~~~~
# These functions are designed to help confrom the unstructured data output by
# find_physical onto structed data arrays by using scipy's implementation
# of Qhull (Delaunay triangulation)

def _find_integermultiple(x: float, dx: float, tol: float = 1e-6) -> int:
  """
  Find integer multiple of dx close to x within the interval (0,x).

  Returns an integer, q, where q*dx is the largest value within (0,x),
  give or take floating point tolerance.
  if x is positive, returns a less positive number i.e. smaller number.
  if x is negative, returns a less negative number i.e. greater number.
  """
  if not isinstance(x,float):
    raise TypeError("x must be a float")
  elif math.isnan(x):
    return float('nan')
  elif not isinstance(dx,float):
    raise TypeError("dx must be a float")
  elif not isinstance(tol,float):
    raise TypeError("tol must be a float")
  
  q, r = divmod(abs(x),dx)
  
  if abs(r-dx)<tol:
      q += 1
  if x<0:
      q = -q
  
  return int(q)


def find_interpolation(xdata: np.ndarray, ydata: np.ndarray, dx: float,
                                                tol: float = 1e-6) -> tuple :
  """
  Interpolates xdata and ydata onto a structured grid with triangulation.
  
  Given xdata and ydata are 2D arrays where the first index is uniformly
  sampled, and the second index in non-uniformly sampled (as generally output
  by find_physical), return uniformly sampled data based on linear
  interpolation by Delaunay triangulation.

  Raises a TypeError or ValueError for some inputs. Raises AssertionError if
  the interpolation fails.
  """
  if not isinstance(xdata,np.ndarray):
    raise TypeError("xdata must a numpy array")
  elif not (len(xdata.shape)==2):
    raise ValueError("xdata must have two dimensions")
  elif not isinstance(ydata,np.ndarray):
    raise TypeError("ydata must be a numpy array")
  elif not (len(ydata.shape)==2):
    raise ValueError("ydata must have two dimensions")
  elif not (xdata.shape==ydata.shape):
    raise ValueError("xdata and ydata must be the same shape")
  elif not isinstance(dx,float):
    raise TypeError("dx must be a float")
  elif not (dx > 0):
    raise ValueError("dx must be a positive float")
  elif not isinstance(tol,float):
    raise TypeError("tol must be a float")
  elif not (tol > 0):
    raise ValueError("tol must be a positive float")
  
  from scipy.interpolate import griddata
  
  t = np.repeat(
        np.arange(ydata.shape[0])[:,None],ydata.shape[1],axis=-1).ravel()
  x = xdata.ravel()
  y = ydata.ravel()
  
  for a in [t,x,y]:
    assert np.isfinite(a).any(), "invalid data"
  
  ti = np.arange(ydata.shape[0])
  
  P = _find_integermultiple(np.nanmin(xdata),dx)
  Q = _find_integermultiple(np.nanmax(xdata),dx)
  
  xi = np.arange(P,Q+1,1)
  
  locs = (ti[:,None],dx*xi[None,:])
  
  # t, x, and y must all be finite for valid index
  mask = np.isfinite(np.stack((t,x,y),axis=-1)).all(axis=-1)
  points = (t[mask], x[mask])
  values = y[mask]
  
  assert mask.any(), "something has gone wrong..."
  
  ynew = griddata(points,values,locs,method='linear',rescale=True)
  
  # ynew lies on a convex hull from griddata
  # at the sides of the images there may be columns with
  # some nan values, however there will be some interior square
  # of the convex hull with no nan values
  # there is probably a better way to find this square, but
  # here we assume that cropping on just the x axis is enough
  mask = np.isfinite(ynew).all(axis=0)
  xi_start, xi_stop = xi[mask].min(), xi[mask].max()
  xi = np.arange(xi_start,xi_stop+1,1)
  ynew = ynew[:,mask]
  
  assert np.isfinite(ynew).all(), "still some nan values..."
  
  return xi.ravel(), ynew, dx


#~~~~~~~~~~~~~~~~~~HELPER FUNCTIONS FOR COMBINING DATA~~~~~~~~~~~~~~~~~~
# These functions are designed to help merge LIF measurements from two
# or more overlapping locations into a single data set.

def find_adjustment(tdata : tuple, xdata : tuple, ydata : tuple,
                    numstept=10,numstepx=10,tol=1e-6) -> tuple:
  """
  Find best fit of data with temporal and spatial offset in range. Returns
  the tuple err, dt, dx.

  Finds a temporal and spatial offset to apply to the temporal and spatial
  locations of the lif data such that the corresponding elevation data has
  minimal absolute difference. find_adjustment takes a brute force approach,
  and will compare the difference in ydata at overlapping tdata and xdata
  locations for all offsets within plus or minus numstept and numstepx. By
  default 400 possible offsets are evaluated. tdata and xdata must be
  integer types in order to find the overlapping tdata and xdata locations.

  Raises a TypeError for some inputs. Raises a ValueError if there is no
  intersection in tdata & xdata,
  """
  if not (isinstance(tdata,tuple) and len(tdata)==2):
    raise TypeError("tdata must be a tuple with length 2")
  elif not (tdata[0].dtype==int and tdata[1].dtype==int):
    raise TypeError(f"t in tdata must have dtype int but has dtypes " \
                                    f"{tdata[0].dtype} and {tdata[1].dtype}")
  elif not (isinstance(xdata,tuple) and len(xdata)==2):
    raise TypeError("xdata must be a tuple with length 2")
  elif not (xdata[0].dtype==int and xdata[1].dtype==int):
    raise TypeError(f"x in xdata must have dtype int but has dtypes " \
                                    f"{xdata[0].dtype} and {xdata[1].dtype}")
  elif not (isinstance(ydata,tuple) and len(ydata)==2):
    raise TypeError("ydata must be a tuple with length 2")
  
  # create all possibile pairs of offsets in the range
  if numstept == 0:
    dt = np.asarray([0],dtype=int)
  else:
    dt = np.arange(-numstept,numstept+1)
  
  if numstepx == 0:
    dx = np.asarray([0],dtype=int)
  else:
    dx = np.arange(-numstepx,numstepx+1)
  
  DT, DX = tuple(np.meshgrid(dt,dx))
  pos = np.transpose(np.stack([DT.ravel(),DX.ravel()]))
  
  # for each possible offset in space and time, estimate the error
  err = np.empty(DT.ravel().shape)
  err[:] = np.nan # invalid by default
  for idx, p in enumerate(pos):
    dt, dx = p
    _, tidx0, tidx1 = np.intersect1d(tdata[0],tdata[1]+dt,return_indices=True)
    _, xidx0, xidx1 = np.intersect1d(xdata[0],xdata[1]+dx,return_indices=True)
    
    # it is possible that dt and dx will push them out of overlapping
    # skip in that case (err[idx] = np.nan by default)
    if not (   tidx0.size==0 or xidx0.size==0
            or tidx1.size==0 or xidx1.size==0 ):
      yidx0 = tuple(np.meshgrid(tidx0,xidx0,indexing = 'ij'))
      yidx1 = tuple(np.meshgrid(tidx1,xidx1,indexing = 'ij'))
      
      #err[idx] = np.mean(np.abs(ydata[0][yidx0] - ydata[1][yidx1]))
      err[idx] = np.mean((ydata[0][yidx0] - ydata[1][yidx1])**2)

  # error out if there is no intersection of the data for any offset
  if np.isnan(err).all():
    raise ValueError("xdata and tdata have no intersection")

  idx_min = np.nanargmin(err)
  dt, dx = pos[idx_min]
  
  return err[idx_min], dt, dx


def find_weightedoverlap(tdata : tuple,xdata : tuple,ydata : tuple) -> tuple:
  """
  Finds a weighted average of elevation data where there is overlap. Returns
  the tuple yidx0, yidx1, ylap.
  
  Finds a weighted average of elevation data where the temporal and spatial
  data overlap. The weights vary linearly on the spatial axis from each end
  of the intersection. Requires temporal and spatial data are provided in
  integer format (e.g. the integer n where the assocaited time is n*dt).
  """
  if   not isinstance(tdata,tuple):
    raise TypeError("tdata must be a tuple")
  elif not isinstance(xdata,tuple):
    raise TypeError("xdata must be a tuple")
  elif not isinstance(ydata,tuple):
    raise TypeError("ydata must be a tuple")
  elif not (len(tdata) == len(xdata) == len(ydata) == 2):
    raise ValueError("tdata, xdata, and ydata must have len of two")
  elif not (len(tdata[0].shape) == 1 and len(tdata[1].shape) == 1):
   raise ValueError("each item in tdata must have one axis")
  elif not (len(xdata[0].shape) == 1 and len(xdata[1].shape) == 1):
    raise ValueError("each item in xdata must have one axis")
  elif not (len(ydata[0].shape) == 2 and len(ydata[1].shape) == 2):
    raise ValueError("each item in ydata must have two axes")
  elif not np.all(np.diff(tdata[0]) > 0) and np.all(np.diff(tdata[1]) > 0):
    raise ValueError("each item in tdata must be monotonically increasing")
  elif not np.all(np.diff(xdata[0]) > 0) and np.all(np.diff(xdata[1]) > 0):
    raise ValueError("each item in xdata must be monotonically increasing")
  elif not (xdata[0].min() < xdata[1].min() and xdata[0].max() < xdata[1].max()):
    raise ValueError("xdata[0] must start and end lower in value than xdata[1]")
  
  # Assume uniformly sampled in both time and space
  # Assume tdata and xdata are integer type arrays for both items in tuple
  _, tidx0, tidx1 = np.intersect1d(tdata[0],tdata[1],return_indices=True)
  _, xidx0, xidx1 = np.intersect1d(xdata[0],xdata[1],return_indices=True)
  
  yidx0 = tuple(np.meshgrid(tidx0,xidx0,indexing = 'ij'))
  yidx1 = tuple(np.meshgrid(tidx1,xidx1,indexing = 'ij'))
  
  P, Q = len(xdata[0][xidx0]), len(tdata[0][tidx0])
  
  w0 = np.repeat(np.linspace(1,0,P).reshape(1,P),Q,axis=0)
  w1 = np.repeat(np.linspace(0,1,P).reshape(1,P),Q,axis=0)
  
  ylap = w0*(ydata[0][yidx0]) + w1*(ydata[1][yidx1])
  
  return yidx0, yidx1, ylap


def list_adjust_data(tdata: list, xdata: list, ydata: list,
                     copy: bool = True, numstept: int = 10,
                     numstepx: int = 10) -> tuple:
  """
  Returns the recommended adjustments for tdata and xdata as
  (adjust_t, adjust_x). By default creates a copy of the data
  to modify, otherwise the input data will be modified in place.
  """
  # Input checking TBD

  # create a copy of each numpy array
  if copy:
    tdata = [t.copy() for t in tdata]
    xdata = [x.copy() for x in xdata]
    ydata = [y.copy() for y in ydata]

  adjust_t = np.zeros((len(ydata)-1,),dtype=int)
  adjust_x = np.zeros((len(ydata)-1,),dtype=int)
  for idx, t1, t2, x1, x2, y1, y2 in zip(range(len(tdata)-1),
                                         tdata[:-1],tdata[1:],
                                         xdata[:-1],xdata[1:],
                                         ydata[:-1],ydata[1:]):
    _, dt, dx = find_adjustment((t1,t2),(x1,x2),(y1,y2),
                                numstept=numstept, numstepx=numstepx)
    adjust_t[idx] = dt
    adjust_x[idx] = dx
    t2 += dt
    dx += dx

  return adjust_t, adjust_x

def list_merge_data(tdata: list, xdata: list, ydata: list,
                    adjust: bool = False) -> tuple:
  """
  Returns the tuple t, x, y where each is a single np.ndarray merged
  from the data provided as a list. If adjust=True, applies an adjustment
  offset to tdata and xdata based on the output of find_adjustment

  

  Raises a TypeError or ValueError for some inputs.
  """
  if   not isinstance(tdata,list):
    raise TypeError("tdata must be a list")
  elif not isinstance(xdata,list):
    raise TypeError("xdata must be a list")
  elif not isinstance(ydata,list):
    raise TypeError("ydata must be a list")
  elif not (len(tdata) == len(xdata) == len(ydata)):
    raise ValueError("tdata, xdata, and ydata must have the same length")
  elif not isinstance(adjust,bool):
    raise TypeError("adjust must be a bool")

  # each element should be a numpy array
  tdata_type = [ isinstance(t,np.ndarray) for t in tdata ]
  xdata_type = [ isinstance(x,np.ndarray) for x in xdata ]
  ydata_type = [ isinstance(y,np.ndarray) for y in ydata ]

  if   not all(tdata_type):
    raise TypeError("all elements in tdata must be np.ndarray")
  elif not all(xdata_type):
    raise TypeError("all elements in xdata must be np.ndarray")
  elif not all(ydata_type):
    raise TypeError("all elements in ydata must be np.ndarray")

  # make sure all y are (N,M), t is (N,) and x is (M,)
  shape_compare = [ y.shape == t.shape + x.shape for t,x,y in zip(tdata,xdata,ydata) ]
  tdata_shape_len = [ len(t.shape)==1 for t in tdata ]
  xdata_shape_len = [ len(x.shape)==1 for x in xdata ]
  ydata_shape_len = [ len(y.shape)==2 for y in ydata ]
  # make sure location data is monotonically increasing with index
  tdata_monotonic = [ np.all(np.diff(t)>0) for t in tdata ]
  xdata_monotonic = [ np.all(np.diff(x)>0) for x in xdata ]

  if   not all(shape_compare):
    raise ValueError("shape must match all data")
  elif not all(tdata_shape_len):
    raise ValueError("each item in tdata must have 1 axis")
  elif not all(xdata_shape_len):
    raise ValueError("each item in xdata must have 1 axis")
  elif not all(ydata_shape_len):
    raise ValueError("each item in ydata must have 2 axes")
  elif not all(tdata_monotonic):
    raise ValueError("each item in tdata must be monotonically increasing")
  elif not all(xdata_monotonic):
    raise ValueError("each item in xdata must be monotonically increasing")

  xdata_min = np.array([ x.min() for x in xdata ],dtype=int)
  xdata_max = np.array([ x.max() for x in xdata ],dtype=int)

  # each item in tdata should overlap but not lie within its neighbors
  # we have already checked that tdata is mononotically increasing for each
  # item in the list, now we must sort the list by min and max
  # if they are the same sort, then they are increasing but do not lie
  # within each other (they might not overlap at this point)
  xdata_min_sortidx = np.argsort(xdata_min)
  xdata_max_sortidx = np.argsort(xdata_max)
  if not (xdata_min_sortidx == xdata_max_sortidx).all():
    raise ValueError("some xdata lies entirely within another, all xdata" \
                     "must have some unique measurements")

  # sort all by increasing xdata
  sortidx = xdata_min_sortidx
  tdata = [ tdata[idx] for idx in sortidx ]
  xdata = [ xdata[idx] for idx in sortidx ]
  ydata = [ ydata[idx] for idx in sortidx ]

  # now that they are sorted in increasing order, ensure that each
  # overlaps with the next 
  xdata_overlapping = np.greater(xdata_max[sortidx][:-1],xdata_min[sortidx][1:]).all()
  if not xdata_overlapping:
    raise ValueError("not all xdata are overlapping when sorted")

  # this may not be enough checks for data that is not uniformly sampled
  # these checks appear to be enought for data with a step size of 1, i.e.
  # all([ np.all(np.diff(t)==1) for t in tdata ]) = True
  # all([ np.all(np.diff(x)==1) for x in xdata ]) = True
  # so there may be some edge cases that are not tested for if the above
  # conditions are not true

  if adjust:
    for t1, t2, x1, x2, y1, y2 in zip(tdata[:-1],tdata[1:],xdata[:-1],xdata[1:],
                                      ydata[:-1],ydata[1:]):
      _, dt, dx = find_adjustment((t1,t2),(x1,x2),(y1,y2))
      # assumes that t2 and x2 are references to the arrays so that they may be
      # modified in place without fancy footwork in the zip
      t2 += dt
      x2 += dx

  # now find a time array that intersects with all tdata
  time = np.intersect1d(tdata[0],tdata[1])
  for t in tdata[2:]:
    time = np.intersect1d(t,time)

  if time.size == 0:
    raise ValueError("there is no overlap in tdata")

  # reduce data in lists to intersect time exclusively
  for idx,t,y in zip(range(len(tdata)),tdata,ydata):
    _, tidx, _ = np.intersect1d(t,time,return_indices=True)
    tdata[idx] = t[tidx]
    ydata[idx] = y[tidx,:]

  # replace ydata in overlapping regions
  for idx in range(len(tdata)-1):
    yidx1, yidx2, ylap = find_weightedoverlap((tdata[idx],tdata[idx+1]),
                                              (xdata[idx],xdata[idx+1]),
                                              (ydata[idx],ydata[idx+1]))
    ydata[idx  ][yidx1] = ylap
    ydata[idx+1][yidx2] = ylap

  # combine xdata and ydata into a single array by appending non-overlapping
  # data. Here is is assumed that the overlapping data was included in the
  # perviously appended data and that all data is on the same time for axis 0
  space = xdata[0]
  elevation = ydata[0]
  for x1, x2, _, y2 in zip(xdata[:-1],xdata[1:],ydata[:-1],ydata[1:]):
    _, _, xidx2 = np.intersect1d(x1,x2,return_indices=True)
    
    xmask2 = np.ones(x2.shape,dtype=bool)
    xmask2[xidx2] = False

    space = np.append(space,x2[xmask2])
    elevation = np.append(elevation,y2[:,xmask2],axis=1)

  return time, space, elevation

#~~~~~~~~~~~~~~~~~~HELPER FUNCTIONS FOR ESTIMATING DEPTH~~~~~~~~~~~~~~~~~~
# these funtions are not complete or well tested

def __find_camerapose(img,board_size,square_size,camera_matrix, dist_coefs):
  """ estimate camera pose in the opencv sense"""
  image_points = find_chessboard_points(img,board_size)
  world_points = np.zeros((board_size[0]*board_size[1],3), np.float32)
  world_points[:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1,2)
  
  _, rvecs, t = cv.solvePnP(world_points, image_points, camera_matrix, dist_coefs)
  R, _ = cv.Rodrigues(rvecs)
  
  if R[0][0] < 0:
    theta = 3.14
    Rtemp = np.array([
        [np.cos(theta),-np.sin(theta),0],
        [np.sin(theta),np.cos(theta),0],
        [0,0,1]
        ])
    R = Rtemp.dot(R)
  
  return R,t

def __find_pixelunitvector(u,v,camera_matrix):
  """ estimate unit vector in pixel space """
  fx = camera_matrix[0,0]
  cx = camera_matrix[0,2]
  fy = camera_matrix[1,1]
  cy = camera_matrix[1,2]
  uvec = np.array([(u-cx)/fx,(v-cy)/fy,1])
  return uvec/np.linalg.norm(uvec)


def __apply_snellslaw(ivec,nvec,n1,n2):
  """ estimate refraction from snells law  """
  assert type(ivec) is np.ndarray and ivec.shape == (3,) \
      and np.isfinite(ivec).all() \
      and type(nvec) is np.ndarray and nvec.shape == (3,) \
      and np.isfinite(nvec).all() \
      and type(n1) is float and type(n2) is float, 'invalid input'

  mu = n1/n2
  n_dot_i = nvec.dot(ivec)
  tvec = np.sqrt(1-((mu**2)*(1 - n_dot_i**2)))*nvec \
         + mu*(ivec-(n_dot_i*nvec))
  assert check_snellslaw(ivec,tvec,nvec,n1,n2), f"invalid input with i: {ivec[0]}, j: {ivec[1]}, k:{ivec[2]}"
  return tvec


def __check_snellslaw(ivec,tvec,nvec,n1,n2,tol=1e-6):
  """ check if vector is estimate of snells law"""
  assert type(ivec) is np.ndarray and ivec.shape == (3,) \
      and np.isfinite(ivec).all() \
      and type(tvec) is np.ndarray and tvec.shape == (3,) \
      and np.isfinite(tvec).all() \
      and type(nvec) is np.ndarray and nvec.shape == (3,) \
      and np.isfinite(nvec).all() \
      and type(n1) is float and type(n2) is float and type(tol) is float, 'invalid input'

  return (np.cross(nvec,tvec) - (n1/n2)*np.cross(nvec,ivec) < tol).all() \
            and np.abs(np.linalg.norm(ivec) - 1) < tol \
            and np.abs(np.linalg.norm(tvec) - 1) < tol


def __find_realdepth(s,ivec,nvec,n1,n2):
  """ estimate real depth from apparent depth """
  assert type(s) is float, 'invalid input'
  assert type(ivec) is np.ndarray and ivec.shape == (3,) \
      and ~(~np.isfinite(ivec)).any(), f'invalid input with: {ivec}'
  assert type(nvec) is np.ndarray and nvec.shape == (3,) \
      and ~(~np.isfinite(nvec)).any(), 'invalid input'
  assert type(n1) is float and type(n2) is float, 'invalid input'

  tvec = apply_snellslaw(ivec,nvec,n1,n2)
  i2 = ivec[1]
  i3 = ivec[2]
  t2 = tvec[1]
  t3 = tvec[2]
  return s*(i3/t3)*(t2/i2)


def __find_realdepth_vector(s,ivec,nvec,n1,n2):
  """ estimate the real depth from apparent for each element of array """
  assert type(s) is np.ma.MaskedArray and type(ivec) is np.ndarray \
      and type(nvec) is np.ndarray and nvec.shape == (3,) \
      and type(n1) is float and type(n2) is float, 'invalid input'
  real_depth = np.empty(s.shape)
  for idx in range(len(s)):
    if (type(s.mask) is np.ndarray and s.mask[idx]) \
            or ~(np.isfinite(ivec[:,idx]).all()):
      real_depth[idx] = np.nan
      continue

    real_depth[idx] = find_realdepth(float(s[idx]),ivec[:,idx],nvec,n1,n2)

  return real_depth

def __pixel_vec_array(surface,camera_matrix):
  """ find pixel vectors """
  assert type(surface) is np.ma.MaskedArray and len(surface.shape) == 1
  u_bot, v_bot, w_bot = pixelvector_from_vdata(surface.reshape(1,-1))
  u_bot_vec = np.empty((3,u_bot.shape[1]))
  for i in range(u_bot.shape[1]):
    if type(surface.mask) is np.ndarray and surface.mask[i]:
      u_bot_vec[:,i] = np.nan
      continue
    u_bot_vec[:,i] = find_pixelunitvector(u_bot[0,i],v_bot[0,i],camera_matrix)

  return u_bot_vec


def __estimate_depth(surface_air,surface_bot,camera_matrix,H,R_pixel2calib,
                     n_air,n_wat):
  """ estimate water depth from measured water surfaces """
  assert type(surface_air) is np.ma.MaskedArray \
    and type(surface_bot) is np.ma.MaskedArray \
    and len(surface_air.shape) == 1 and len(surface_bot.shape) == 1 \
    and type(H) is np.ndarray and H.shape == (3,3) \
    and type(R_pixel2calib) is np.ndarray and R_pixel2calib.shape == (3,3) \
    and is_rotationmatrix(R_pixel2calib) \
    and type(n_air) is float and type(n_wat) is float, 'invalid input'

  # find physical requires at least 2 in time, simply repeat to work around
  air_tmp = np.repeat(surface_air.reshape(1,-1),2,axis=0)
  bot_tmp = np.repeat(surface_bot.reshape(1,-1),2,axis=0)
  x_air, eta_air, R_calib2world = find_physical(H,air_tmp,zero=False)
  _    , eta_bot, _ = find_physical(H,bot_tmp,R=R_calib2world,zero=False)
  apparent_depth = np.abs(eta_air[0,:]) - np.abs(eta_bot[0,:])

  # somehow surface_bot is backwards?
  u_bot_vec = pixel_vec_array(surface_bot[::-1],camera_matrix)

  # calib2world from the homography does not seem to be relevant
  # to slope errors however it does account for positive/negative depth
  R_calib2world = np.transpose(R_calib2world)

  ivec = R_calib2world.dot(R_pixel2calib.dot(u_bot_vec))
  nvec = np.array([0,1,0])

  return find_realdepth_vector(apparent_depth,ivec,nvec,n_air,n_wat)


