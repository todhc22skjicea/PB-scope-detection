# PB-scope-detection

## Introduction

![Framework of PB-scope-detection](https://github.com/todhc22skjicea/PB-scope-detection/blob/main/pb_detection/quantification_testing/PBdetection_framework.png)

PB-scope-detection is a target detection project based on YOLOv7, focusing on the analysis of p-body images. 
Before starting, ensure that you have the following dependencies installed:
Python 3.8 or higher
PyTorch 1.7 or higher
CUDA (if using a GPU)
Other necessary dependencies (such as OpenCV, NumPy, etc.)
See [Official YOLOv7](https://github.com/WongKinYiu/yolov7)

## Dataset preparation

The dataset for P-body screening are now available at https://zenodo.org/records/15202103

## Training

To train the model, use the following command:
```
python train.py --workers 8 --device 0 --batch-size 32 --data data/molecule.yaml --img 128 128 --cfg /training/yolov7.yaml --weights 'yolov7_training.pt' --name molecule625 --hyp data/hyp.scratch.custom.yaml
```

## Testing

To perform target detection on a single image, use the following command:
```
python detect.py 
```
For batch testing of P-bodies images, you can first use [Cellpose](https://github.com/MouseLand/cellpose) to obtain npy binary segmentation masks(neccesary for brightness quantification). After modifying the necessary parameters and file paths in the script, run the following command:
```
python detectpbody.py
```
The Imagej macro used for threshold batch processing in this paper could be found at ['\preprocessing\preprocessing.ijm'](https://github.com/todhc22skjicea/PB-scope-detection/tree/main/pb_detection/preprocessing)

## Performance
![Perfomance of PB-scope-detection](https://github.com/todhc22skjicea/PB-scope-detection/blob/main/pb_detection/quantification_testing/PBdetection.png)

The result of this detection method acheived >95% agreement with manual annotations labeled by two experts working in the related fields.
The raw images label and detection result can be accessed [quantification_testing](https://github.com/todhc22skjicea/PB-scope-detection/tree/main/pb_detection/quantification_testing).

## Citation
If you use this repository, please cite:
```
@inproceedings{
}
```

## License

This project is licensed under the GNU License. See LICENSE for details.

## Acknowledgement

[Official YOLOv7](https://github.com/WongKinYiu/yolov7)

We thank the authors of the above for making their code public.
