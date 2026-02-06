import cv2 as cv
from ultralytics import YOLO
model = YOLO('best.pt')

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret != True: break

    #frame = cv.imread("Bloks_3.jpg")
    frame = cv.resize(frame, (640, 480))
    result = model(frame)[0]
    annotated_frame = result.plot(
        conf=True,  # не показывать уверенность
        boxes=True,  # показывать bounding boxes
        labels=True,  # показывать метки классов
        probs=True,  # не показывать вероятности классов
    )

    cv.imshow("result", annotated_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Очистка
# cap.release()
cv.destroyAllWindows()

