import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import threading

engine = pyttsx3.init()

def speak_text(text):
    """Function to speak the text asynchronously."""
    engine.say(text)
    engine.runAndWait()

# Load the trained model
model_dict = pickle.load(open('./model_last_2.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Detect hands with a confidence threshold
hands = mp_hands.Hands(min_detection_confidence=0.6, max_num_hands=1)

prev_frame_time = 0
new_frame_time = 0
threshold = 8

ratio =0.8
spaceWord=0
thresholdSpace = 10

cache = []
text = '' 

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
text_color = (255, 255, 255) 

while True:
    data_aux = []
    x_ = []
    y_ = []
    z_ = []

    ret, frame = cap.read()
    # if not ret:
    #     break  # Thoát nếu không lấy được khung hình

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(frame_rgb)

    # Only proceed if a hand is detected
    if results.multi_hand_landmarks:
        spaceWord = 0
        # Chọn bàn tay đầu tiên
        hand_landmarks = results.multi_hand_landmarks[0]  # Chỉ lấy bàn tay đầu tiên

        # Vẽ landmarks lên frame
        mp_drawing.draw_landmarks(
            frame,  # image to draw
            hand_landmarks,  # model output
            mp_hands.HAND_CONNECTIONS,  # hand connections
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Xử lý bàn tay đầu tiên
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            z = hand_landmarks.landmark[i].z

            x_.append(x)
            y_.append(y)
            z_.append(z)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            z = hand_landmarks.landmark[i].z
            
            diagonal = (x**2+y**2)**(1/2)

            data_aux.append((x - min(x_))/diagonal)
            data_aux.append((y - min(y_))/diagonal)
            data_aux.append((z - min(z_))/diagonal)

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10


        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10
        
        # Make a prediction
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = prediction[0]

        cache.append(predicted_character)

        if len(cache) >= threshold:
            counter = {}
            for i in cache:
                if i in counter:
                    counter[i]+=1
                else:
                    counter[i]=1
            frequence = max(counter.values())
            if frequence >= ratio* threshold:
                character = max(counter, key=counter.get)
                print(character)   
            cache =[]
            
            if text == '':
                text += character
            else:
                if text[-1] != character:
                    text= text+str(character)
                    print(text)
        

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    elif text != '':
        cache = []
        character = ''
        spaceWord +=1
        if spaceWord > thresholdSpace and text[-1]!=' ':
            text += ' '

            word = text.split(' ')
            threading.Thread(target=speak_text, args=(word[-2],), daemon=True).start()
            print(word[-2])
            print(text)
    
    # Get the dimensions of the frame
    frame_height, frame_width = frame.shape[:2]

    # Calculate the size of the text
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

    # Calculate the position to center-align the text at the bottom
    x = (frame_width - text_width) // 2  # Center horizontally
    y = frame_height - baseline - 10  # Bottom with some padding (10 pixels)

    # Put the text on the frame
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)


    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # Convert FPS to an integer
    fps = int(fps)

    # Put the FPS on the frame
    cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    # Kiểm tra nếu cửa sổ đã bị đóng
    # Chờ 1 mili giây và kiểm tra nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
        break


# Giải phóng tài nguyên và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
