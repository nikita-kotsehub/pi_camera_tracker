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

# modification of he set_servos method from 
# https://pyimagesearch.com/2019/04/01/pan-tilt-face-tracking-with-a-raspberry-pi-and-opencv/        
def set_servos(correct_angle, objX, centerX):
    servo=Servo()
    servo.setup()
    # signal trap to handle keyboard interrupt
    signal.signal(signal.SIGINT, signal_handler)
    # loop indefinitely
    while True:
        # if the correction angle exceeds 15 in either direction, rotate
        if ((90 - correct_angle.value) - servo.prev >= 15) or ((90 - correct_angle.value) - servo.prev <= -15): 
            servo.changeState(90 - correct_angle.value)
        
    servo.destroy()
    
# modification of the run code from 
# https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/raspberry_pi
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

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Initialize the object detection model
  options = ObjectDetectorOptions(
      num_threads=num_threads,
      score_threshold=0.3,
      max_results=3,
      enable_edgetpu=enable_edgetpu)
  detector = ObjectDetector(model_path=model, options=options)

  # find the center of the frame, the desired output value
  _, frame = cap.read()
  (H, W) = frame.shape[:2]
  centerX.value = W//2

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
    
    # check if any of the detection containes a 'person'
    # if yes, update the central location of the person
    for d in detections:
        if d.categories[0].label == 'person':
            objX.value = d.coordinates[0]

    # Draw keypoints and edges on input image
    image = utils.visualize(image, detections)

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
      # use Manager to manage multiple threads and share values
      with Manager() as manager:
          # the correcting angle
          correct_angle = manager.Value("i", 0)
          
          # the X centers of the fram and the object, initial values
          centerX = manager.Value("i", 320)
          objX = manager.Value("i", 320)
          
          # PID values
          panP = manager.Value("f", 0.1)
          panI = manager.Value("f", 0.0004)
          panD = manager.Value("f", 0.000001)
        
          # define the processes
          processRun = Process(target=run, args=(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
              int(args.numThreads), bool(args.enableEdgeTPU), objX, centerX))
          processPid = Process(target=pid_process, args=(correct_angle, panP, panI, panD, objX, centerX))
          processServo = Process(target=set_servos, args=(correct_angle, objX, centerX))
          
          # start the processes
          processRun.start()
          processPid.start()
          processServo.start()
          
          # join the processes
          processRun.join()
          processPid.join()
          processServo.join()
          
  except KeyboardInterrupt:
      servo.destroy()
    
if __name__ == '__main__':
  main()
