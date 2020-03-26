import os
import cv2
import time
from datetime import datetime
import numpy as np
import tensorflow as tf

from utils.app_utils import FPS, WebcamVideoStream, HLSVideoStream
from multiprocessing import Queue, Pool
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
now = datetime.now()
dt_string = now.strftime("%d_%m_%YT%H%M%S")
print("date and time =", dt_string)
# Initialize webcam feed
video = cv2.VideoCapture('Workout Buddies App Commercial (30 Sec) (1).mp4')
#ret = video.set(3,1280)
#ret = video.set(4,720)
# the output will be written to output.avi
out = cv2.VideoWriter(
    'output_video/'+dt_string+'.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))

#captureimage
def image_capture(f):
    cv2.imwrite('out_image/'+dt_string+'.jpg',f)
    return 2
z=1
mins=0

while(True):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    #calling image caputre func
    if(z<=1):
        z=image_capture(frame)
        
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    
    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60)
    #print(name)
    # All the results have been drawn on the frame, so it's time to display it.
#    out.write(frame.astype('uint8'))
    cv2.imshow('Object detector', frame)
    #time.sleep(1)
    #mins += 1
    # Press 'q' to quit
    key=cv2.waitKey(1)
    if key==27:
        break

# Clean up
video.release()
cv2.destroyAllWindows()


