# PBDetection
## Introduction
PBDetection is a target detection project based on YOLOv7, focusing on the analysis of p-body images. 
Before starting, ensure that you have the following dependencies installed:
Python 3.8 or higher
PyTorch 1.7 or higher
CUDA (if using a GPU)
Other necessary dependencies (such as OpenCV, NumPy, etc.)
See[Official YOLOv7](https://github.com/WongKinYiu/yolov7)
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
For batch testing of P-bodies images, you can first use [Cellpose](https://github.com/MouseLand/cellpose) to obtain npy binary segmentation masks. After modifying the necessary parameters and file paths in the script, run the following command:
```
python detectpbody.py
```
The Imagej macro used for threshold batch processing in this paper could be found at '\preprocessing\preprocessing.ijm'
## Citation
If you use this repository, please cite:
```
@inproceedings{
}
```

## License

This project is licensed under the MIT License. See LICENSE for details.

## Acknowledgement

[Official YOLOv7](https://github.com/WongKinYiu/yolov7)

We thank the authors of the above for making their code public.