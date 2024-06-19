import cv2
import numpy as np
import time
import winsound
import logging
import ctypes
import serial
from threading import Thread, Lock
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FallDetectionSystem:
    def __init__(self, yolo_weights, yolo_cfg, bed_region_image, serial_port, baud_rate, speed_threshold=200):
        self.net, self.output_layers = self.load_yolo_model(yolo_weights, yolo_cfg)
        self.bed_region = self.detect_bed_region(bed_region_image)
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.speed_threshold = speed_threshold
        self.person_positions = deque(maxlen=2)
        self.video_stream = VideoStreamHandler().start()
        self.feedback_data = {"false_positives": 0, "true_positives": 0}

    def load_yolo_model(self, weights, config):
        try:
            net = cv2.dnn.readNet(weights, config)
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
            logging.info("YOLO model loaded successfully.")
            return net, output_layers
        except Exception as e:
            logging.error(f"Error loading YOLO model: {e}")
            return None, None

    def detect_persons(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:  # Person class
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        persons = [boxes[i] for i in indexes]
        return persons

    def calculate_speed(self, pos1, pos2, time_interval):
        _, y1, _, h1 = pos1
        _, y2, _, h2 = pos2
        distance = abs((y2 + h2 / 2) - (y1 + h1 / 2))  # Vertical distance between the centers
        speed = distance / time_interval
        return speed

    def is_fall_detected(self, current_position, time_interval):
        if len(self.person_positions) < 2:
            return False

        prev_position = self.person_positions[-1]
        _, prev_y, _, prev_h = prev_position
        _, y, w, h = current_position

        bed_x, bed_y, bed_w, bed_h = self.bed_region

        # Check if the person is outside the bed region
        if not (bed_x <= current_position[0] <= bed_x + bed_w and
                bed_y <= current_position[1] <= bed_y + bed_h):
            # Check for a significant downward movement
            if y > prev_y + prev_h * 0.5:  # Person has moved significantly downwards
                return True

            # Check if the aspect ratio suggests a fallen person
            aspect_ratio = h / (w + 1e-5)
            if aspect_ratio < 0.5:  # Bounding box is wide, suggesting a horizontal position
                return True

            # Calculate speed of movement to detect sudden drops
            speed = self.calculate_speed(prev_position, current_position, time_interval)
            logging.info(f"Speed: {speed}")
            if speed > self.speed_threshold:  # Adjust threshold based on empirical data
                return True

        return False

    def send_desktop_notification(self):
        try:
            message = "Alert: Fall detected!"
            title = "Fall Detection Alert"
            ctypes.windll.user32.MessageBoxW(0, message, title, 0x40 | 0x1)
            logging.info("Desktop notification sent!")
        except Exception as e:
            logging.error(f"Failed to send desktop notification: {e}")

    def play_audio_alert(self):
        try:
            winsound.PlaySound('alert_sound.wav', winsound.SND_FILENAME)
            logging.info("Audio alert played!")
        except Exception as e:
            logging.error(f"Failed to play audio alert: {e}")

    def send_bluetooth_alert(self):
        try:
            with serial.Serial(self.serial_port, self.baud_rate, timeout=1) as ser:
                message = "Alert: Fall detected!"
                ser.write(message.encode())
                logging.info("Bluetooth alert sent!")
        except serial.SerialException as e:
            logging.error(f"Failed to send Bluetooth alert: {e}")

    def get_user_feedback(self):
        feedback = input("Was this a fall? (yes/no): ").strip().lower()
        if feedback == 'yes':
            self.feedback_data["true_positives"] += 1
        elif feedback == 'no':
            self.feedback_data["false_positives"] += 1

    def adjust_detection_thresholds(self):
        false_positives = self.feedback_data["false_positives"]
        true_positives = self.feedback_data["true_positives"]

        if false_positives + true_positives == 0:
            return 200  # Default speed threshold

        false_positive_rate = false_positives / (false_positives + true_positives)
        if false_positive_rate > 0.3:  # Example threshold
            new_speed_threshold = min(300, 200 * (1 + false_positive_rate))
            logging.info(f"Adjusting speed threshold to: {new_speed_threshold}")
            return new_speed_threshold
        return 200

    def detect_bed_region(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Error: Could not open or read the image at {image_path}")
            return None

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 150, 0])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

        points = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            points.append((x, y))
            points.append((x + w, y))
            points.append((x, y + h))
            points.append((x + w, y + h))

        if len(points) < 4:
            logging.error("Error: Not enough points detected to form the bed trapezium.")
            return None

        points = sorted(points, key=lambda p: (p[1], p[0]))
        bed_trapezium = [points[0], points[1], points[-1], points[-2]]
        bed_x = min(point[0] for point in bed_trapezium)
        bed_y = min(point[1] for point in bed_trapezium)
        bed_w = max(point[0] for point in bed_trapezium) - bed_x
        bed_h = max(point[1] for point in bed_trapezium) - bed_y

        return (bed_x, bed_y, bed_w, bed_h)

    def run(self):
        if self.net is None or self.output_layers is None or self.bed_region is None:
            logging.error("Initialization failed. Exiting.")
            return

        time.sleep(2.0)  # Allow camera sensor to warm up

        prev_time = time.time()
        while True:
            current_time = time.time()
            if current_time - prev_time > 60:  # Example maximum duration for processing
                break

            frame = self.video_stream.read()
            persons = self.detect_persons(frame)
            self.speed_threshold = self.adjust_detection_thresholds()
            logging.info(f"Current speed threshold: {self.speed_threshold}")

            if persons:
                current_position = persons[0]  # Assuming only one person in frame
                self.person_positions.append(current_position)

                if self.is_fall_detected(current_position, current_time - prev_time):
                    logging.info("Fall detected!")
                    Thread(target=self.send_desktop_notification).start()
                    Thread(target=self.play_audio_alert).start()
                    Thread(target=self.send_bluetooth_alert).start()
                    self.get_user_feedback()

                prev_time = current_time

            for (x, y, w, h) in persons:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            bed_x, bed_y, bed_w, bed_h = self.bed_region
            cv2.rectangle(frame, (bed_x, bed_y), (bed_x + bed_w, bed_y + bed_h), (0, 0, 255), 2)
            cv2.imshow("Fall Detection", frame)

            if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
                break

        self.video_stream.stop()
        cv2.destroyAllWindows()

class VideoStreamHandler:
    def __init__(self):
        self.stopped = False
        self.lock = Lock()
        self.cap = cv2.VideoCapture(0)
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Initialize with an empty frame

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            ret, frame = self.cap.read()
            if not ret:
                logging.error("Error: Could not read frame.")
                self.stop()
                return
            with self.lock:
                self.frame = cv2.resize(frame, (640, 480))  # Resize frame to reduce computation

    def read(self):
        with self.lock:
            frame = self.frame.copy()
        return frame

    def stop(self):
        self.stopped = True
        self.cap.release()

if __name__ == "__main__":
    yolo_weights = "yolov3.weights"
    yolo_cfg = "yolov3.cfg"
    bed_region_image = 'C:/Users/dhruv/Downloads/bed_trapezium_detected.jpg'
    serial_port = "COM3"  # Replace with your serial port
    baud_rate = 9600

    fall_detection_system = FallDetectionSystem(yolo_weights, yolo_cfg, bed_region_image, serial_port, baud_rate)
    fall_detection_system.run()
    