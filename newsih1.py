import cv2
import numpy as np
import dlib
import os
import face_recognition

def load_known_faces(known_faces_dir="captured_images"):
    if not os.path.exists(known_faces_dir):
        print(f"Directory {known_faces_dir} does not exist. Please check the path.")
        return [], []

    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(known_faces_dir):
        if filename.endswith(('.jpg', '.png')):
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)[0]

            known_face_encodings.append(face_encoding)
            known_face_names.append(name)

    return known_face_encodings, known_face_names

def verify():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    known_face_encodings, known_face_names = load_known_faces()

    def calculate_relative_movement(landmarks, prev_landmarks):
        movement = np.linalg.norm(landmarks - prev_landmarks, axis=1)
        movement_ratios = movement / (np.linalg.norm(landmarks, axis=1) + 1e-5)
        return movement, movement_ratios

    def detect_3d_liveliness(frame, prev_landmarks=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        for face in faces:
            shape = predictor(gray, face)
            landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

            if prev_landmarks is not None:
                movement, movement_ratios = calculate_relative_movement(landmarks, prev_landmarks)
                mean_movement = np.mean(movement)
                max_movement_ratio = np.max(movement_ratios)

                movement_threshold = .525
                movement_ratio_threshold = 0.0238

                blink_detected = False  

                if mean_movement < movement_threshold and max_movement_ratio < movement_ratio_threshold and not blink_detected:
                    cv2.putText(frame, "2D Image Detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "3D Face Detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Analyzing...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            for (x, y) in landmarks:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

            return frame, landmarks

        return frame, None

    def detect_phone(frame):
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        phone_detected = False

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if class_id == classes.index("cell phone") and confidence > 0.5:
                    phone_detected = True
                    break

        return phone_detected

    def recognize_face(frame, known_face_encodings, known_face_names):
        rgb_frame = frame[:, :, ::-1]  
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        name = "Unknown"

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        return name

    def process_frame(frame, prev_landmarks):
        phone_detected = detect_phone(frame)
        if phone_detected:
            cv2.putText(frame, "Spoofing Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            prev_landmarks = None  
        else:
            name = recognize_face(frame, known_face_encodings, known_face_names)
            frame, prev_landmarks = detect_3d_liveliness(frame, prev_landmarks)
            cv2.putText(frame, name, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame, prev_landmarks

    cam = cv2.VideoCapture(0)

    prev_landmarks = None

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame, prev_landmarks = process_frame(frame, prev_landmarks)

        cv2.imshow('3D Liveliness and Spoof Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    verify()
