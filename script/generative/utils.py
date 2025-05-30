import numpy as np
import pydicom
from  PIL import Image

def load_image(image_path, percentile=None, clahe=False):
    dcm = pydicom.dcmread(image_path)
    # invert pixel array if necessary
    if dcm.PhotometricInterpretation == 'MONOCHROME1':
        pixel_array = np.invert(dcm.pixel_array)
    else:
        pixel_array = dcm.pixel_array
    # convert to uint8
    # dcm.pixel_array = dcm.pixel_array.astype(np.uint8)
    if clahe:
        from skimage import exposure
        pixel_array = exposure.equalize_adapthist(pixel_array, clip_limit=0.01)
        pixel_array = (pixel_array * 255)
    if percentile is not None:
        return normalize(pixel_array, percentile=percentile)
    return pixel_array

def normalize(x, wlww=None, minmax=None, percentile=None):
    if wlww is None and minmax is None and percentile is None:
        raise RuntimeError('Either wlww or minmax is required')

    if wlww is not None:
        wl, ww = wlww
        minmax = (wl-ww/2, wl+ww/2)

    if percentile is not None:
        minmax = np.percentile(x,[percentile,100-percentile])

    x = np.clip(x, minmax[0], minmax[1]).astype(np.float32)
    return np.round(255*(x - minmax[0])/(minmax[1]-minmax[0])).astype(np.uint8)

def save_as_img(filename, image):
    # convert to uint8
    image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(filename)