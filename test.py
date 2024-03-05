import cv2
import numpy as np
from keras.models import load_model

# Load pre-trained model
model = load_model(r'D:\AI_PROJECT\CNN_FACE\model\Final_test1.h5')

# Load haarcascade for face detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

classes = {0:'chunghoang',2:'thaongo',1:'duycao',3:'vandung'}
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        status = 'no face'
    else:
        for x, y, w, h in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            faces = faceCascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in faces:
                face_roi = roi_color[ey: ey + eh, ex:ex + ew]
                new_array = cv2.resize(face_roi, (150, 150))
                X_input = np.array(new_array).reshape(-1, 150, 150, 3).astype('float64')
                Predict = model.predict(X_input)
                predicted_class = np.argmax(Predict)
                if 0 <= predicted_class < len(classes):
                    status = classes[predicted_class]
                else:
                    status = "Không nhận diện được khuôn mặt"

                # Vẽ tên của người được nhận diện cạnh khuôn mặt
                cv2.putText(frame, f'{status} ', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12),
                            2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
