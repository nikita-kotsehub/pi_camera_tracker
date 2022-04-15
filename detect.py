# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import time

from multiprocessing import Manager
from multiprocessing import Process
import signal

import cv2
from object_detector import ObjectDetector
from object_detector import ObjectDetectorOptions
import utils

from servo import Servo
from PID import PID
# set PID values for panning
#P = 0.09
#I = 0.08
#D = 0.002

    
# function to handle keyboard interrupt
def signal_handler(sig, frame):
    # print a status message
    print("[INFO] You pressed `ctrl + c`! Exiting...")
    # disable the servo
    servo.destroy()
    # exit
    sys.exit()
    
def pid_process(output, p, i, d, objCoord, centerCoord):
    # signal trap to handle keyboard interrupt
    signal.signal(signal.SIGINT, signal_handler)
    # create a PID and initialize it
    pid = PID(p.value, i.value, d.value)
    pid.initialize()
    # loop indefinitely
    while True:
        # calculate the error
        error = centerCoord.value - objCoord.value
        # update the value
        output.value = pid.update(error)
        #print('from PID:', output.value)
        
def set_servos(correct_angle, rn, objX, centerX):
    servo=Servo()
    servo.setup()
    # signal trap to handle keyboard interrupt
    signal.signal(signal.SIGINT, signal_handler)
    # loop indefinitely
    while True:
        # if the pan angle is within the range, pan
        if ((90 - correct_angle.value) - servo.prev >= 15) or ((90 - correct_angle.value) - servo.prev <= -15): 
            print(f'Goal-Angle:{90 - correct_angle.value}, Corretion: {correct_angle.value}, Diff: {centerX.value - objX.value}')
            servo.changeState(90 - correct_angle.value)
        #time.sleep(0.2)
        #print(f'Goal-Angle:{90 + correct_angle.value}, Corretion: {correct_angle.value}, Diff: {centerX.value - objX.value}')
    servo.destroy()
    
def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool, objX, centerX) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  options = ObjectDetectorOptions(
      num_threads=num_threads,
      score_threshold=0.3,
      max_results=3,
      enable_edgetpu=enable_edgetpu)
  detector = ObjectDetector(model_path=model, options=options)

  #pid = PID()
  #pid.initialize()
  _, frame = cap.read()

  (H, W) = frame.shape[:2]
  x_central = 0.5
  centerX.value = W//2
  #y_central = H // 2
  #central_coords=(x_central, y_central)
  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    image = cv2.flip(image, 1)

    # Run object detection estimation using the model.
    detections = detector.detect(image)
    for d in detections:
        if d.categories[0].label == 'person':
            #print(d.bounding_box)
            #print(d.coordinates)
            objX.value = d.coordinates[0]
            #print(objX.value, centerX.value)
            #error = central_coords[0] - d.coordinates[0]
            #correct_angle = pid.update(error)
            #print(correct_angle)
            
            #print(g"{d.categories[0].label}")
            #servoWrite(30)
    # Draw keypoints and edges on input image
    image = utils.visualize(image, detections)

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    #fps_text = 'FPS = {:.1f}'.format(fps)
    #text_location = (left_margin, row_size)
    #cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                #font_size, text_color, font_thickness)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('object_detector', image)

  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0_edgetpu.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=True)
  args = parser.parse_args()
  try:
      with Manager() as manager:
        # correct_angle values will be managed by independed PIDs

          correct_angle = manager.Value("i", 90)
          
          centerX = manager.Value("i", 320)
          objX = manager.Value("i", 320)
          
          panP = manager.Value("f", 0.1)
          panI = manager.Value("f", 0.0004)
          panD = manager.Value("f", 0.000001)
        
          processRun = Process(target=run, args=(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
              int(args.numThreads), bool(args.enableEdgeTPU), objX, centerX))
          processPid = Process(target=pid_process, args=(correct_angle, panP, panI, panD, objX, centerX))
          processServo = Process(target=set_servos, args=(correct_angle, True, objX, centerX))
          #run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
              #int(args.numThreads), bool(args.enableEdgeTPU))
          
          processRun.start()
          processPid.start()
          processServo.start()
          
          processRun.join()
          processPid.join()
          processServo.join()
          
          servo.destroy()
          
  except KeyboardInterrupt:
      servo.destroy()
    
if __name__ == '__main__':
  main()
