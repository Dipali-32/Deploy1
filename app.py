from flask import Flask, render_template, Response, stream_with_context
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time
import json

app = Flask(__name__)

# Load model and labels
model = tf.keras.models.load_model("cnn_normal_pose_classifier.h5")
class_labels = [
    "adho mukh svanasana",
    "ashtanga namaskara",
    "ashwa sanchalanasana",
    "bhujangasana",
    "hasta utthanasana",
    "kumbhakasana",
    "padahastasana",
    "pranamasana"
]
pose_status = {label: False for label in class_labels}

# Load MoveNet
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = movenet.signatures['serving_default']

POSE_CONNECTIONS = [
    (0, 1), (1, 3), (0, 2), (2, 4), 
    (5, 7), (7, 9), (6, 8), (8, 10), 
    (5, 6), (5, 11), (6, 12), 
    (11, 13), (13, 15), (12, 14), (14, 16), 
    (11, 12)
]

def draw_landmarks(img, keypoints, threshold=0.3):
    h, w, _ = img.shape
    for i, (y, x, c) in enumerate(keypoints):
        if c > threshold:
            cv2.circle(img, (int(x * w), int(y * h)), 5, (0, 255, 0), -1)
    for (i, j) in POSE_CONNECTIONS:
        y1, x1, c1 = keypoints[i]
        y2, x2, c2 = keypoints[j]
        if c1 > threshold and c2 > threshold:
            pt1 = (int(x1 * w), int(y1 * h))
            pt2 = (int(x2 * w), int(y2 * h))
            cv2.line(img, pt1, pt2, (255, 0, 0), 2)

def gen_frames():
    cap = cv2.VideoCapture(0)
    conf_threshold = 0.8
    kp_threshold = 0.3
    min_valid_kp = 15

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_input = tf.image.resize_with_pad(tf.convert_to_tensor(img_rgb), 192, 192)
        input_tensor = tf.expand_dims(tf.cast(img_input, dtype=tf.int32), axis=0)

        outputs = movenet(input_tensor)
        keypoints = outputs['output_0'].numpy()[0, 0, :, :]
        valid_kp = np.sum(keypoints[:, 2] > kp_threshold)

        label = ""

        if valid_kp >= min_valid_kp:
            input_data = keypoints.flatten().reshape(1, 17, 3, 1)
            prediction = model.predict(input_data, verbose=0)
            pred_class = np.argmax(prediction)
            confidence = np.max(prediction)

            if confidence > conf_threshold:
                label = class_labels[pred_class]
                pose_status[label] = True  # Mark as detected
                  
        draw_landmarks(frame, keypoints)

        if label:
            cv2.putText(frame, f"{label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', labels=class_labels)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pose_updates')
def pose_updates():
    def event_stream():
        while True:
            time.sleep(1)
            yield f"data: {json.dumps(pose_status)}\n\n"
    return Response(stream_with_context(event_stream()), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)


# from flask import Flask, render_template, Response
# import cv2
# import numpy as np
# import tensorflow as tf
# import tensorflow_hub as hub
# import threading
# import time
# import pyttsx3

# app = Flask(_name_)

# # Load CNN model and labels
# model = tf.keras.models.load_model("cnn_pose_classifier.h5")
# class_labels = [
#     "adho mukh svanasana",
#     "ashtanga namaskara",
#     "ashwa sanchalanasana",
#     "bhujangasana",
#     "hasta utthanasana",
#     "kumbhakasana",
#     "padahastasana",
#     "pranamasana"
# ]

# # Load MoveNet
# movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4").signatures['serving_default']

# POSE_CONNECTIONS = [
#     (0, 1), (1, 3), (0, 2), (2, 4), 
#     (5, 7), (7, 9), (6, 8), (8, 10), 
#     (5, 6), (5, 11), (6, 12), 
#     (11, 13), (13, 15), (12, 14), (14, 16), 
#     (11, 12)
# ]

# # Timer state
# current_pose = None
# pose_start_time = None
# pose_timer_active = False

# # Voice engine setup
# engine = pyttsx3.init()

# def speak(text):
#     def run():
#         engine.say(text)
#         engine.runAndWait()
#     threading.Thread(target=run).start()

# def draw_landmarks(img, keypoints, threshold=0.3):
#     h, w, _ = img.shape
#     for i, (y, x, c) in enumerate(keypoints):
#         if c > threshold:
#             cv2.circle(img, (int(x * w), int(y * h)), 5, (0, 255, 0), -1)
#     for (i, j) in POSE_CONNECTIONS:
#         y1, x1, c1 = keypoints[i]
#         y2, x2, c2 = keypoints[j]
#         if c1 > threshold and c2 > threshold:
#             pt1 = (int(x1 * w), int(y1 * h))
#             pt2 = (int(x2 * w), int(y2 * h))
#             cv2.line(img, pt1, pt2, (255, 0, 0), 2)

# def gen_frames():
#     global current_pose, pose_start_time, pose_timer_active
#     cap = cv2.VideoCapture(0)
#     conf_threshold = 0.8
#     kp_threshold = 0.3
#     min_valid_kp = 15

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         img_input = tf.image.resize_with_pad(tf.convert_to_tensor(img_rgb), 192, 192)
#         input_tensor = tf.expand_dims(tf.cast(img_input, dtype=tf.int32), axis=0)

#         outputs = movenet(input_tensor)
#         keypoints = outputs['output_0'].numpy()[0, 0, :, :]
#         valid_kp = np.sum(keypoints[:, 2] > kp_threshold)

#         label = ""
#         if valid_kp >= min_valid_kp:
#             input_data = keypoints.flatten().reshape(1, 17, 3, 1)
#             prediction = model.predict(input_data, verbose=0)
#             pred_class = np.argmax(prediction)
#             confidence = np.max(prediction)

#             if confidence > conf_threshold:
#                 label = class_labels[pred_class]

#                 if label != current_pose:
#                     current_pose = label
#                     pose_start_time = time.time()
#                     pose_timer_active = True
#                     speak(f"{label} is detected and let's do this for 5 minutes.")
#             elif confidence <= conf_threshold:
#                 label = ""

#         if pose_timer_active and time.time() - pose_start_time >= 300:
#             speak(f"{current_pose} is completed.")
#             pose_timer_active = False

#         draw_landmarks(frame, keypoints)
#         cv2.putText(frame, f"{label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
#                     (0, 255, 0) if label else (0, 0, 255), 3)

#         _, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('indexd.html', poses=class_labels)

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if _name_ == '_main_':
#     app.run(debug=True)
# from flask import stream_with_context

# # @app.route('/pose_event')
# # def pose_event():
# #     def stream():
# #         last_pose = None
# #         while True:
# #             if current_pose != last_pose:
# #                 last_pose = current_pose
# #                 yield f"data: {current_pose}\n\n"
# #             time.sleep(1)
# #     return Response(stream_with_context(stream()), mimetype='text/event-stream')

# from flask import stream_with_context

# pose_timer_active = False
# pose_start_time = None
# current_pose = None

# @app.route('/pose_event')
# def pose_event():
#     def stream():
#         global current_pose
#         last_pose = None
#         while True:
#             if current_pose != last_pose:
#                 last_pose = current_pose
#                 yield f"data: {current_pose}\n\n"
#             time.sleep(1)
#     return Response(stream_with_context(stream()), mimetype='text/event-stream')