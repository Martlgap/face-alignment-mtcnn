import facealignment
import cv2
import numpy as np
import pytest


@pytest.mark.singleface
def test_single_face():
    tool = facealignment.FaceAlignmentTools(weights_path="weights/")
    img = cv2.imread('tests/samples/single_face.png')
    assert type(tool.align(img)) == np.ndarray
    assert type(tool.align(img, allow_multiface=True)) == list
    assert len(tool.align(img, allow_multiface=True)) == 1


@pytest.mark.multiface
def test_multi_face():
    tool = facealignment.FaceAlignmentTools(weights_path="weights/")
    img = cv2.imread('tests/samples/multi_face.png')
    assert type(tool.align(img)) == np.ndarray
    assert len(tool.align(img, allow_multiface=True)) == 2
    assert type(tool.align(img, allow_multiface=True)) == list


@pytest.mark.noface
def test_no_face():
    tool = facealignment.FaceAlignmentTools(weights_path="weights/")
    img = cv2.imread('tests/samples/no_face.png')
    assert tool.align(img) is None
    assert tool.align(img, allow_multiface=True) is None
