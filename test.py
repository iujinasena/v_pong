import numpy as np
import cv2 as cv
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#initial-racket-body-XY
cx = 300
cy = 240

#initial-racket-computer
rcx1 = 300
rcx2 = 291
rc_side = 1

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 400)
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue
    image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image_height, image_width, _ = image.shape
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        for ids, landmrk in enumerate(hand_landmarks.landmark):
            if(ids==9):
                cx, cy = landmrk.x * image_width, landmrk.y*image_height
                print(ids, cx, cy)

    #background
    img = np.zeros((400,600,3), np.uint8)

    #outline
    cv.rectangle(img,(0,0),(599,599),(255,255,255),1)

    #board
    tlh = np.array([[205,100],[140,240],[460,240],[395,100]], np.int32)

    cv.line(img,(205,99),(395,99),(225,225,225),2)#top-line
    cv.line(img,(205,100),(140,240),(225,225,225),3)#left-line
    cv.line(img,(395,100),(460,240),(225,225,225),3)#right-line

    tlh = np.array([[206,101],[142,238],[458,238],[394,101]], np.int32)
    cv.fillPoly(img, pts = [tlh], color = (32,80,30))#table

    cv.line(img,(300,101),(300,240),(225,225,225),3)#divider
    cv.line(img,(140,240),(460,240),(225,225,225),3)#bottom-line
    cv.line(img,(140,244),(460,244),(60,60,60),3)#table-edge

    #net-shadow
    tlh = np.array([[192,133],[180,157],[422,157],[410,133]], np.int32)
    cv.fillPoly(img, pts = [tlh], color = (28,54,28))
    cv.line(img,(300,133),(300,157),(160,160,160),3)

    #net
    for i in range(161,440,3):
        cv.line(img,(i,130),(i,157),(0,30,0),1)#net-v
    for i in range(131,157,3):
        cv.line(img,(160,i),(440,i),(0,30,0),1)#net-h

    cv.line(img,(160,130),(440,130),(255,255,255),2)#net-top

    if(rc_side == 1):
      rcx1 +=2 
      rcx2 +=2
    else:
      rcx1 -=2
      rcx2 -=2

    if(rcx1 == 394):
      rc_side = 0
    if(rcx1 == 206):
      rc_side = 1

    rcx3 = rcx2+18

    #racket-computer
    cv.circle(img,(rcx1,50), 18, (26,18,164), -1)#racket-body
    cv.circle(img,(rcx1,50), 19, (62,171,250), 1)#racket-body
    cv.line(img,(rcx1,68),(rcx1,86),(62,171,250),5)#racket-hand
    cv.line(img,(rcx2,66),(rcx3,66),(62,171,250),2)#racket-hand

    #ball
    cv.circle(img,(282,190), 6, (240,240,240), -1)#ball-body
    cv.circle(img,(282,190), 6, (20,20,20), 1)#ball-body

    #racket-user
    cx = int(cx)
    cy = int(cy)
    cv.circle(img,(cx,cy), 32, (26,18,164), -1)#racket-body
    cv.circle(img,(cx,cy), 32, (62,171,250), 1)#racket-body
    cv.line(img,(cx,cy+30),(cx,cy+55),(62,171,250),7)#racket-hand
    cv.line(img,(cx-11,cy+28),(cx+11,cy+28),(62,171,250),3)#racket-hand

    cv.imshow("V Pong", img)

    if cv.waitKey(1) & 0xFF == 27:
        cv.destroyAllWindows()

cap.release()