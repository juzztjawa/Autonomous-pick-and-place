import sys
import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
import processing
# Function for stereo vision and depth estimation
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import pyrealsense2 as rs
from sklearn.decomposition import PCA
from PIL import Image
import time

def Convert_2Dto3D(point,depth_intrin,depth_image):
    distance = depth_image[point[1], point[0]]
    depth_point=rs.rs2_deproject_pixel_to_point(depth_intrin, point, distance)
    return depth_point

sys.path.append("..")
sam_checkpoint = "D:\\jawahar\\iiit classnotes\\5th sem\\Grid\\sam_vit_l_0b3195.pth"
model_type = "vit_l"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
depth_sensor = pipeline_profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    color_image=color_image[40:360,120:540]
    depth_image=depth_image[40:360,120:540]
    print(depth_image.shape)
    masks=processing.process(color_image,mask_generator)
    lower=200
    upper=400

    # for mask in masks:
    #     if(mask["area"]>0):
    #         maskofInt=mask['segmentation']
    #         break
    maskofInt=masks[5]['segmentation']

    # arr=masks[1]["segmentation"]
    data=Image.fromarray(maskofInt)
    np_arr=np.array(data)
    box_segment = np.zeros(maskofInt.shape[:2], dtype=np.uint8)
    for i in range(len(maskofInt)):
        for j in range(len(maskofInt[i])):
            if(maskofInt[i][j]==False):
                box_segment[i,j]=0
            else:
                box_segment[i,j]=255

    segmented_image=processing.show_anns(masks)
    box_segment_rgb = cv2.cvtColor(box_segment, cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(box_segment_rgb, 0.7, segmented_image, 0.3, 0)
    cv2.imshow('Annotations', overlay)
    cv2.imwrite('maskseg.png',overlay)
    box_segment_rgb=processing.contour(box_segment_rgb)
    cv2.imshow('Mask with Contours',box_segment_rgb)
    # mask_arr=np.array(maskofInt['segmentation'])
    pointcloud=np.multiply(maskofInt,depth_image)#test

    #pca

    pca = PCA(n_components=3)
    pca.fit(pointcloud)

    orientation = pca.components_

    roll = np.degrees(np.arctan2(orientation[2, 1], orientation[2, 2]))
    pitch = np.degrees(np.arctan2(-orientation[2, 0], np.sqrt(orientation[2, 1]**2 + orientation[2, 2]**2)))
    yaw = np.degrees(np.arctan2(orientation[1, 0], orientation[0, 0]))

    #euler to quaternion
    quaternion=processing.euler_to_quaternion(yaw,pitch,roll)
    #perpendicular
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)
    yaw_rad = np.radians(yaw)

    # Define the reference direction (forward direction)
    reference_direction = np.array([1, 0, 0])

    # Create rotation matrices for pitch, roll, and yaw
    rotation_matrix_pitch = np.array([[1, 0, 0],
                                    [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
                                    [0, np.sin(pitch_rad), np.cos(pitch_rad)]])

    rotation_matrix_roll = np.array([[np.cos(roll_rad), 0, np.sin(roll_rad)],
                                    [0, 1, 0],
                                    [-np.sin(roll_rad), 0, np.cos(roll_rad)]])

    rotation_matrix_yaw = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                                    [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                                    [0, 0, 1]])

    # Combine the rotation matrices (order matters, typically yaw-roll-pitch)
    combined_rotation_matrix = np.dot(rotation_matrix_yaw, np.dot(rotation_matrix_roll, rotation_matrix_pitch))

    # Apply the combined rotation to the reference direction to get the approach direction
    approach_direction = np.dot(combined_rotation_matrix, reference_direction)

    # print(f"Approach direction: {approach_direction}")
    x_coord,y_coord=processing.get_xy(maskofInt)[1]
    #depth_image*mask
    point=(x_coord,y_coord)
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    
    World_point=Convert_2Dto3D(point,depth_intrin,depth_image)
    distance=World_point[2]
    
    cv2.circle(color_image, point, 4, (0, 0, 255))
    cv2.putText(color_image, "{}mm".format(distance), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    # inverted_depth_image = 255 - depth_image
    inverted_depth_image=processing.normalize(depth_image)
    #cv2.imshow("depth frame", inverted_depth_image)

    # Normalize the depth map to the 0-255 range
    depth_map_normalized = ((depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()) * 255).astype(np.uint8)

    # Create a grayscale image
    depth_image_color = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

    # Show the depth image
    cv2.imshow('Depth Map', depth_image_color)

    cv2.imshow("Color frame", color_image)
    print("World point:- ",World_point)
    print("Quarternion:- ",quaternion)
    print("Approach direction:-",approach_direction)
    # time.sleep(10)
    key = cv2.waitKey(1)
    if key == 27:
        break