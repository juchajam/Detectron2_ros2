import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node



def generate_launch_description():
    pkg_detectron2_ros = get_package_share_directory('detectron2_ros')

    # Input parameters declaration
    input_topic = LaunchConfiguration('input_topic')
    detection_threshold = LaunchConfiguration('detection_threshold')
    config = LaunchConfiguration('config')
    model = LaunchConfiguration('model')
    visualization = LaunchConfiguration('visualization')

    declare_input_topic_cmd = DeclareLaunchArgument(
        'input_topic',
        default_value='/tv_image',
        description='image topic name')

    declare_detection_threshold_cmd = DeclareLaunchArgument(
        'detection_threshold',
        default_value='0.5',
        description='threshold to filter the detection results [0, 1]')
    
    declare_config_cmd = DeclareLaunchArgument(
        'config',
        default_value=os.path.join(pkg_detectron2_ros[:pkg_detectron2_ros.find('install')], 'src', 'Detectron2_ros2', 'detectron2', 'configs', 'COCO-InstanceSegmentation', 'mask_rcnn_R_50_FPN_3x.yaml'),
        description='path to the config file.')
    
    declare_model_cmd = DeclareLaunchArgument(
        'model',
        default_value='detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl',
        description='path to the training model file. For example: /detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
    
    declare_visualization_cmd = DeclareLaunchArgument(
        'visualization',
        default_value='True',
        description='True or False to pubish the result like a image')
    
    detectron2_ros_node = Node(
        package='detectron2_ros',
        executable='detectron2_ros',
        parameters=[
            {'input_topic': input_topic},
            {'detection_threshold': detection_threshold},
            {'config': config},
            {'model': model},
            {'visualization': visualization},
        ],
        output='screen'
    )
        
    ld = LaunchDescription()

    ld.add_action(declare_input_topic_cmd)
    ld.add_action(declare_detection_threshold_cmd)
    ld.add_action(declare_config_cmd)
    ld.add_action(declare_model_cmd)
    ld.add_action(declare_visualization_cmd)
    
    ld.add_action(detectron2_ros_node)

    return ld
