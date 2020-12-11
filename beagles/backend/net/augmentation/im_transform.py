import numpy as np
import cv2
import sys
from functools import partial
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
import albumentations as A

all_pixel_transforms = [cls.__name__ for cls in ImageOnlyTransform.__subclasses__()]
all_spatial_transforms = [cls.__name__ for cls in DualTransform.__subclasses__()]


class RandomChannelAmplify(ImageOnlyTransform):

	def __init__(self, alpha=0.1, always_apply=False, p=0.5):
		super(RandomChannelAmplify, self).__init__(always_apply, p)
		self.alpha = alpha
		t = [np.random.uniform()]
		t += [np.random.uniform()]
		t += [np.random.uniform()]
		self.t = np.array(t) * 2.0 - 1.0

	def get_transform_init_args(self):
		return {'alpha': self.alpha}

	def get_transform_init_args_names(self):
		return ('alpha',)

	def get_params_dependent_on_targets(self, params):
		pass

	def apply(self, image, alpha=0.1, **params):
		# random amplify each channel
		im = image * (1 + self.t * alpha)
		mx = 255.0 * (1 + alpha)
		up = np.random.uniform() * 2 - 1
		im = cv2.pow(im / mx, 1.0 + up * 0.5)
		return np.array(im * 255.0, np.uint8)


def pixel_transform(image: np.ndarray, *args):
	args = filter(lambda i: i in all_pixel_transforms, args)
	get_concrete = partial(getattr, A)
	args = [cls() for cls in list(map(get_concrete, args))]
	transform = A.Compose([RandomChannelAmplify(), *args])
	return transform(image=image)

def spatial_transform(image: np.ndarray, annotation, *args):
	"""Scale and translate image"""
	args = filter(lambda i: i in all_spatial_transforms, args)
	get_concrete = partial(getattr, A)
	args = [cls() for cls in list(map(get_concrete, args))]
	class_labels = [i[0] for i in annotation]
	annotation = [i[-4:] for i in annotation]
	bbox_params = A.BboxParams(format='pascal_voc', label_fields=['class_labels'])
	transform = A.Compose([A.RandomScale(), A.HorizontalFlip(), *args], bbox_params=bbox_params)
	transformed = transform(image=image, bboxes=annotation, class_labels=class_labels)
	bboxes = [[class_labels[i], *box] for i, box in enumerate(transformed["bboxes"])]
	return transformed["image"], bboxes
