import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np


ZONE_POLYGON = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8m.pt")

    box_annotator = sv.BoxAnnotator(thickness=6)
    label_annotator = sv.LabelAnnotator(text_thickness=4, text_scale=2)

    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon)
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.RED,
        thickness=2,
        text_thickness=4,
        text_scale=2
    )
    # frame_id = 0
    while True:
        ret, frame = cap.read()

        # frame_id += 1
        # if frame_id % 3 != 0:
        #     continue

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_ultralytics(result)
        # detections = detections[detections.class_id == 0]
        zone.trigger(detections=detections)
        labels = [
            f"{result.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _
            in detections
        ]
        

        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(annotated_frame, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
        annotated_frame = zone_annotator.annotate(annotated_frame)  
        
        cv2.imshow("yolov8", annotated_frame)

        if (cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()