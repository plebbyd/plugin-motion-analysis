import cv2

cap = cv2.VideoCapture('pond_001.avi')
width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    
    cv2.imshow('window', frame)
    cv2.waitKey(20)