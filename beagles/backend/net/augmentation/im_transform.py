import numpy as np
import cv2


def imcv2_recolor(image, alpha=.1):
	t = [np.random.uniform()]
	t += [np.random.uniform()]
	t += [np.random.uniform()]
	t = np.array(t) * 2. - 1.

	# random amplify each channel
	im = image * (1 + t * alpha)
	mx = 255. * (1 + alpha)
	up = np.random.uniform() * 2 - 1
# 	im = np.power(im/mx, 1. + up * .5)
	im = cv2.pow(im/mx, 1. + up * .5)
	return np.array(im * 255., np.uint8)


def imcv2_affine_trans(image: np.ndarray):
	"""Scale and translate image"""
	h, w, c = image.shape
	scale = np.random.uniform() / 10. + 1.
	max_offx = (scale-1.) * w
	max_offy = (scale-1.) * h
	offx = int(np.random.uniform() * max_offx)
	offy = int(np.random.uniform() * max_offy)
	
	image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
	image = image[offy: (offy + h), offx: (offx + w)]
	flip = np.random.binomial(1, .5)
	if flip:
		image = cv2.flip(image, 1)
	return image, [w, h, c], [scale, [offx, offy], flip]
