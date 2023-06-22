import cv2
import numpy as np
from yolov7 import Yolov7
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

class YOLOv7DeepSORTVideoInference:
    def __init__(self, model_weights, video_path, output_path):
        self.yolo = Yolov7(model_weights)
        self.tracker = Tracker(nn_budget=100)
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, self.fourcc, self.fps, (self.width, self.height))
        self.num_people = 0
    
    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            detections = []
            class_ids, scores, boxes = self.yolo.detect(frame)
            if class_ids is not None:
                for i, box in enumerate(boxes):
                    if class_ids[i] == 0: # sadece insanlar
                        detections.append(Detection(box, scores[i], None))
            
            features = self.yolo.encoder(frame, boxes)
            detections = [Detection(bbox, confidence, feature) for bbox, confidence, feature in zip(boxes, scores, features)]
            self.tracker.predict()
            self.tracker.update(detections)

            self.num_people = 0
            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                cv2.putText(frame, "ID: {}".format(str(track.track_id)), (int(bbox[0]), int(bbox[1]) - 10), 0, 1, (0, 255, 0), 2)
                self.num_people += 1
            
            # Kişi sayısı
            cv2.putText(frame, "Kişi Sayısı: {}".format(str(self.num_people)), (10, 30), 0, 1, (0, 0, 255), 2)

            self.out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
