import RPi.GPIO as GPIO
import time

class Servo:
    def __init__(self):
        self.OFFSE_DUTY = 0.5        #define pulse offset of servo
        self.SERVO_MIN_DUTY = 2.5+self.OFFSE_DUTY     #define pulse duty cycle for minimum angle of servo
        self.SERVO_MAX_DUTY = 12.5+self.OFFSE_DUTY    #define pulse duty cycle for maximum angle of servo
        self.servoPin = 12
        self.prev=90

    def map_vals(self, value, fromLow, fromHigh, toLow, toHigh):  # map a value from one range to another range
        return (toHigh-toLow)*(value-fromLow) / (fromHigh-fromLow) + toLow

    def setup(self):
        GPIO.setmode(GPIO.BOARD)         # use PHYSICAL GPIO Numbering
        GPIO.setup(self.servoPin, GPIO.OUT)   # Set servoPin to OUTPUT mode
        GPIO.output(self.servoPin, GPIO.LOW)  # Make servoPin output LOW level

        self.p = GPIO.PWM(self.servoPin, 50)     # set Frequece to 50Hz
        self.p.start(0)                     # Set initial Duty Cycle to 0
        self.servoWrite(90)
        
    def servoWrite(self, angle):      # make the servo rotate to specific angle, 0-180 
        if(angle<10):
            angle = 10
        elif(angle > 170):
            angle = 170
        self.p.ChangeDutyCycle(self.map_vals(angle,10,170,self.SERVO_MIN_DUTY,self.SERVO_MAX_DUTY)) # map the angle to duty cycle and output it
            
    def changeState(self, new):
        self.prev = int(self.prev)
        new = int(new)

        if new > self.prev:
            for dc in range(self.prev, new, 1):   # make servo rotate from 0 to 180 deg
                self.servoWrite(dc)     # Write dc value to servo
                time.sleep(0.01) # speed
        elif new < self.prev:
            for dc in range(self.prev, new, -1):   # make servo rotate from 0 to 180 deg
                self.servoWrite(dc)     # Write dc value to servo
                time.sleep(0.01) # speed
        self.prev = new
           
    def destroy(self):
        self.p.stop()
        GPIO.cleanup()