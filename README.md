# face-alignment-mtcnn
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Contributors](https://img.shields.io/github/contributors/martlgap/face-alignment-mtcnn?color=green)](https://img.shields.io/github/contributors/martlgap/face-alignment-mtcnn?color=green)
[![Last Commit](https://img.shields.io/github/last-commit/martlgap/face-alignment-mtcnn)](https://img.shields.io/github/last-commit/martlgap/face-alignment-mtcnn)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://img.shields.io/badge/license-MIT-blue)
[![Downloads](https://img.shields.io/github/downloads/martlgap/face-alignment-mtcnn/total)](https://img.shields.io/github/downloads/face-alignment-mtcnn/total)
[![CI testing](https://github.com/martlgap/face-alignment-mtcnn/actions/workflows/testing.yml/badge.svg)](https://github.com/martlgap/face-alignment-mtcnn/actions/workflows/testing.yml)
[![CI make_release](https://github.com/martlgap/face-alignment-mtcnn/actions/workflows/make_release.yml/badge.svg)](https://github.com/martlgap/face-alignment-mtcnn/actions/workflows/make_release.yml)


## 📘 Description
A very simple and lightweight pure python implementation of face-alignment with [MTCNN](https://arxiv.org/abs/1604.02878) landmark-extractor. The tool uses
tensorflow-lite (CPU only) and supports several platforms. 

Pull request are welcome!


## ⚡️ Features 
- very simple and lightweight face detection and face alignment based on [MTCNN](https://arxiv.org/abs/1604.02878)
- no GPU needed
- multi platform support (Windows, MacOS, Ubuntu)


## ✅ ToDos
- [ ] Warnings if image will be heavily distorted during alignment process


## 🥣 Requirements
- [Python 3.8](https://www.python.org/)
- [TensorflowLite-Runtime 2.5.0](https://www.tensorflow.org/lite/guide/python)
- Tested on: Ubuntu 20.04, Windows 10, MacOS 11.3


## ⚙️ How to install tflite-runtime
You can easily install tflite-runtime from https://google-coral.github.io/py-repo/ with the following line:
```zsh
pip3 install tflite-runtime --find-links https://google-coral.github.io/py-repo/tflite-runtime
```

## ⚙️ How to install the face-alignment package
Simply install the package via pip from git:
```zsh
pip3 install git+https://github.com/martlgap/face-alignment-mtcnn
``` 
or if you do not have git installed on your system, install it directly from the wheel:
```zsh
pip3 install https://github.com/Martlgap/face-alignment-mtcnn/releases/latest/download/face_alignment_mtcnn-0.1-py3-none-any.whl
``` 


## 🏠 Basic Usage
- Simply import the facealignment package
- Instantiate the tools
- Use the "align()" method to align a face from an arbitrary image
```shell
import facealignment

tools = facealignment.FaceAlignTools()
aligned_face = tools.align(<image>)
```


## 🚀 Run Example
Download repository and run it with Python 3.8:
```shell
python example.py
```
A cv2 window will pop up, showing you the alignment. Press any key, to 
go to the next cv2 window and finally close the window


## 🙏 Acknowledgement
- Thanks to Iván de Paz Centeno for his [implementation](https://github.com/ipazc/mtcnn) 
  of [MTCNN](https://arxiv.org/abs/1604.02878) in [Tensorflow 2](https://www.tensorflow.org/). 
  The MTCNN model weights are taken "as is" from his repository and were converted to tflite-models afterwards.
  
