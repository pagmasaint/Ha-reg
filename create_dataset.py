import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7)

DATA_DIR = './data_test'
# count = 1
data = []
labels = []
notrun = []
for member in os.listdir(DATA_DIR):
    member = os.path.join(DATA_DIR, member)
    for dir_ in os.listdir(member):
        for img_path in os.listdir(os.path.join(member, dir_)):
            # if count ==1:
            #     print(dir_)
            #     print(img_path)
            #     print(os.path.join(DATA_DIR, dir_, img_path))
            #     count +=1

            data_aux = []
            x_ = []
            y_ = []
            z_ = []

            img = cv2.imread(os.path.join(member, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


            results = hands.process(img_rgb)
            # print(len(results.multi_hand_landmarks))

            # for hand_landmarks in results.multi_hand_landmarks:
            #     for i in range(len(hand_landmarks.landmark)):
            #         if i == 0:
            #             print(hand_landmarks)
            #             print(type(hand_landmarks))
            #             print('aaaaaaaaaaaaaaaaaaa')
            #             print(hand_landmarks.landmark[i])
            #             print(i)

            if results.multi_hand_landmarks:
                if len(results.multi_hand_landmarks)==1:
                    for hand_landmarks in results.multi_hand_landmarks:
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

                    data.append(data_aux)
                    labels.append(dir_)

            else:
                print(os.path.join(member, dir_, img_path))

f = open('data_full.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
