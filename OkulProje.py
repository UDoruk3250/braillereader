import cv2
import numpy as np
import pygame
pygame.mixer.init()

def splitBoxes(img):
	rows = np.vsplit(img, 3)
	boxes = []
	for r in rows:
		cols = np.hsplit(r, 2)
		for box in cols:
			boxes.append(box)
	return boxes


cam = cv2.VideoCapture(0)
cnts = []

pts2 = np.float32([[0, 0], [210, 0], [0, 297], [210, 297]])
# HSV VALUES FOR COLOR MASK:
# 69,16,203
# 106,193,255
alt = np.array([80, 0, 233])
ust = np.array([179, 255, 255])
while True:
	success, img = cam.read()
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gri, (5, 5), 1)
	# canny = cv2.Canny(blur,10,50)
	# __, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_TOZERO)
	thresh = cv2.inRange(hsv, alt, ust)
	konturlar, wsd = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	area = [round(cv2.contourArea(c)) for c in konturlar]
	if len(area) > 0:
		buyuk = np.argmax(area)
		enBuyuk = konturlar[buyuk]
		if cv2.contourArea(enBuyuk) > 6000:
			x, y, w, h = cv2.boundingRect(enBuyuk)
			cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)
			pts1 = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
			for i in range(0, 4):
				cv2.circle(img, (pts1[i][0], pts1[i][1]), 8, (0, 0, 255), cv2.FILLED)
				matrix = cv2.getPerspectiveTransform(pts1, pts2)
				warp = cv2.warpPerspective(img, matrix, (210, 297))
				cv2.imshow("Warp", warp)
				griwarp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
				_, warpTh = cv2.threshold(griwarp, 200, 255, cv2.THRESH_BINARY)

				kutular = splitBoxes(warpTh)
				cv2.imshow("Line", kutular[0])
				map = np.zeros((3, 2))
				sira = 0
				sutun = 0
				for image in kutular:
					toplamPixel = cv2.countNonZero(image)
					map[sira][sutun] = toplamPixel
					sutun += 1
					if sutun == 2: sira += 1;sutun = 0

				alfabe = np.zeros((3, 2))
				for t in range(0, 3):
					line = map[t]
					# print("Map: ", map)
					# print("Sum: ", np.sum(map))
					ortalama = np.sum(map) / 6 - 750
					# print("Ortalama: ", ortalama)
					for i, item in enumerate(line):
						alfabe[t][i] = 1 if item < ortalama else 0
				alfabe = alfabe.ravel()

				print(alfabe)
				if np.array_equal(alfabe,[1,0,0,0,0,0]):
					cv2.putText(img,"A",(x+w,y+h+50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),5)
					pygame.mixer.music.load('A.mp3')
					pygame.mixer.music.play()
				if np.array_equal(alfabe, [1, 0, 1, 0, 0, 0]):
					cv2.putText(img, "B", (x + w, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
					pygame.mixer.music.load('B.mp3')
					pygame.mixer.music.play()
				if np.array_equal(alfabe,[1,1,0,0,0,0]):
					cv2.putText(img,"C",(x+w,y+h+50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),5)
					pygame.mixer.music.load('C.mp3')
					pygame.mixer.music.play()
				if np.array_equal(alfabe,[1,1,0,1,0,0]):
					cv2.putText(img,"D",(x+w,y+h+50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),5)
					pygame.mixer.music.load('D.mp3')
					pygame.mixer.music.play()
			# 	print(line)
			# print("--------------------------")
			# countNonZero küçükse marked (siyah) değilse unmarked (beyaz)

	cv2.imshow("Canny", thresh)
	cv2.imshow("Original", img)

	cv2.waitKey(1)
