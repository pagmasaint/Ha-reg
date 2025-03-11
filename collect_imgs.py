import os

import cv2


DATA_DIR = './data_test'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

dataset_size = 100
run = True
ckey = -1

cap = cv2.VideoCapture(0)
while run:
    done = False
    while True:
        ret, frame = cap.read()
        key = cv2.waitKey(1)

        if key !=-1:
            ckey = chr(key)
            if not os.path.exists(os.path.join(DATA_DIR, str(ckey))):
                os.makedirs(os.path.join(DATA_DIR, str(ckey)))

            print(ckey)
            break

        cv2.putText(frame, 'Ready? Press label ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)

        cv2.waitKey(1)
        if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
            run = False
            break

    print('Collecting data for class {}'.format(ckey))

    counter = 0
    while counter < dataset_size and run:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(ckey), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()
