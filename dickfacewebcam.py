import cv2
import numpy as np

background = cv2.VideoCapture(0) 

def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")





penis = cv2.imread("penis.png", cv2.IMREAD_UNCHANGED)
while 1:
    success, img = background.read()
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        overlay_transparent(img,penis, (x+w+w)//2, (y)//2)
        cv2.putText(img, "Dickhead", (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
