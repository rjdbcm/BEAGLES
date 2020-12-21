import re
import numpy as np
from typing import Iterator, Tuple
from beagles.base import *
from functools import partial
from beagles.io.logs import get_logger
import albumentations.augmentations.transforms as A
from albumentations import Compose, BboxParams, BasicTransform

INCOMPATIBLE_AUGMENTATIONS = ['CoarseDropout', 'Crop', 'CropNonEmptyMaskIfExists',
							  'ElasticTransform', 'FromFloat', 'GridDistortion',
							  'GridDropout', 'Lambda', 'LongestMaxSize', 'MaskDropout',
							  'PadIfNeeded', 'RandomCrop', 'RandomCropNearBBox',
							  'RandomResizedCrop', 'RandomSizedBBoxSafeCrop',
							  'RandomSizedCrop', 'Resize', 'SmallestMaxSize', 'ToFloat']

ABBREVIATIONS = ['pca', 'clahe', 'iso', 'rgb']

class Augment:

	def __init__(self, *args: AnyStr):
		"""Image transformation API for data augmentation.

		Args:
			*args: string names of Albumentations spatial and pixel transformations.
		"""
		self.log, self.logfile = get_logger()
		self._spatial = self._init_pipeline(self.fix_names(A.DualTransform, args))
		self._pixel = self._init_pipeline(self.fix_names(A.ImageOnlyTransform, args))
		self.spatial_transform = Compose(self._spatial,
										   bbox_params=BboxParams(
											   format = 'pascal_voc',
											   label_fields = ['class_labels']
										   	   )
										   )
		self.pixel_transform = Compose(self._pixel)

	def __len__(self):
		return len(self.spatial_pipeline + self.pixel_pipeline)

	def __repr__(self):
		msg = str()
		active = self.spatial_transform.transforms.transforms or self.pixel_transform.transforms.transforms
		if self.spatial_transform.transforms.transforms:
			msg += 'Albumentations Spatial Augmentation Pipeline:\n'
			for t in self.spatial_transform.transforms:
				msg += f'\t{t}\n'
		if self.pixel_transform.transforms.transforms:
			msg += 'Albumentations Pixel Augmentation Pipeline:\n'
			for t in self.pixel_transform.transforms:
				msg += f'\t{t}\n'
		if not active:
			msg = 'No active augmentation pipeline'
		return msg

	@property
	def pixel_pipeline(self) -> List[Callable]:
		return self._pixel

	@property
	def spatial_pipeline(self) -> List[Callable]:
		return self._spatial

	def _init_pipeline(self, transforms: Iterator[AnyStr]) -> List[Callable]:
		transforms = list(transforms)
		incompatible = [i for i, v in enumerate(transforms) if v in INCOMPATIBLE_AUGMENTATIONS]
		for idx in incompatible:
			self.log.info(f'Ignoring incompatible transform {transforms.pop(idx)}')
		return [cls() for cls in list(map(partial(getattr, A), transforms))]

	def fix_names(self, cls: Type[BasicTransform], args: Tuple[AnyStr]) -> Iterator:
		return filter(None, map(partial(self._fixup_transform_name, cls), args))

	@staticmethod
	def _fixup_transform_name(cls: BasicTransform, arg: AnyStr) -> Union[AnyStr, None]:
		camelcase = r'((?!^)(?<!_)[A-Z][a-z]+|(?<=[a-z0-9])[A-Z])'
		camel_wb = r'_\1'
		all_transforms = [sub.__name__ for sub in cls.__subclasses__()]
		snake_transforms = [re.sub(camelcase, camel_wb, i).lower() for i in all_transforms]
		lower_transforms = [i.lower() for i in all_transforms]
		if arg in all_transforms:
			return arg
		elif arg in snake_transforms:
			words = [i.upper() if i in ABBREVIATIONS else i.title() for i in arg.split('_')]
			return ''.join(words)
		elif arg in lower_transforms:
			return all_transforms[lower_transforms.index(arg)]
		else:
			return None

	def pixel(self, image: np.ndarray) -> np.ndarray:
		"""Transform image pixels.

		Args:
			image: np.ndarray of shape (rows, cols, channels)

		Returns:
			image transformed by Albumentations defined in the .cfg [net] section.

		"""
		return self.pixel_transform(image=image)['image']


	def spatial(self, image: np.ndarray, annotation: List[AnnotatedBBox]) -> Tuple[
		np.ndarray, List[AnnotatedBBox]]:
		"""Transform image and annotations spatially.

		Args:
			image: np.ndarray of shape (rows, cols, channels)
			annotation: list of annotations in BEAGLES voc format (like Pascal VOC but labels first)

		Returns:
			image transformed by Albumentations defined in the .cfg [net] section.
		"""
		class_labels = [i[0] for i in annotation]
		transformed = self.spatial_transform(image=image,
											 bboxes=[i[-4:] for i in annotation],
											 class_labels=class_labels)
		bboxes = [[class_labels[i], *box] for i, box in enumerate(transformed['bboxes'])]
		return transformed['image'], bboxes
