import SimpleITK as sitk
import numpy as np

class SegmentationGenerator:
    def __init__(self, size = [300, 300, 100], origin = [0, 0, 0], spacing = [1, 1, 1]):
        self.size = size
        self.origin = origin
        self.spacing = spacing

    def generate_circle(self, radius, center):
        image = sitk.Image(self.size, sitk.sitkUInt8)
        image.SetSpacing(self.spacing)
        image.SetOrigin(self.origin)

        circle = sitk.GaussianSource(sitk.sitkUInt8, self.size, radius, center)
        circle = sitk.Cast(circle, sitk.sitkUInt8)

        image += circle

        return image

    def generate_cube(self, size, origin):
        image = sitk.Image(self.size, sitk.sitkUInt8)
        image.SetSpacing(self.spacing)
        image.SetOrigin(self.origin)

        cube = sitk.GaussianSource(sitk.sitkUInt8, self.size, size, origin)
        cube = sitk.Cast(cube, sitk.sitkUInt8)

        image += cube

        return image