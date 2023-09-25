# Detectron2 Object Detector for ROS2

A ROS2(humble) Node for detecting objects using [Detectron2](https://github.com/facebookresearch/detectron2).

## Downloading the Package

Clone the package to the ROS workspace using git tools
```bash
cd <your_workspace>/src
git clone https://github.com/juchajam/Detectron2_ros2.git
cd Detectron2_ros2
git pull --all
git submodule update --init
python -m pip install -e detectron2
```

## Compilation

```bash
cd <your_workspace>
colcon build
source <your_workspace>/install/setup.bash
```

## Running

Running the node with launch
You need to 
```bash
ros2 launch detectron2_ros detectron2_ros.launch.py
```

## Arguments

The following arguments can be set on the `ros2 launch` above.
- `input_topic`: image topic name
- `detection_threshold`: threshold to filter the detection results [0, 1]
- `visualization`: True or False to pubish the result like a image
- `model`: path to the training model file. For example: `/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml`

## Citing Detectron
If you use Detectron2 in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry.

```bash
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```
