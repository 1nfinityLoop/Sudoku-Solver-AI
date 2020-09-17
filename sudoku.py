import cv2
import numpy as np
import operator
from keras.models import load_model
from keras.models import model_from_json
import sudoku_solver as sol

classifier = load_model("./digit_model.h5")

marge = 4
case = 28 + 2 * marge
taille_grille = 9 * case

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
flag = 0
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1080, 620))


while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_grille = None
    maxArea = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area > 25000:
            peri = cv2.arcLength(c, True)
            polygone = cv2.approxPolyDP(c, 0.01 * peri, True)
            if area > maxArea and len(polygone) == 4:
                contour_grille = polygone
                maxArea = area

    if contour_grille is not None:
        cv2.drawContours(frame, [contour_grille], 0, (0, 255, 0), 2)
        points = np.vstack(contour_grille).squeeze()
        points = sorted(points, key=operator.itemgetter(1))
        if points[0][0] < points[1][0]:
            if points[3][0] < points[2][0]:
                pts1 = np.float32([points[0], points[1], points[3], points[2]])
            else:
                pts1 = np.float32([points[0], points[1], points[2], points[3]])
        else:
            if points[3][0] < points[2][0]:
                pts1 = np.float32([points[1], points[0], points[3], points[2]])
            else:
                pts1 = np.float32([points[1], points[0], points[2], points[3]])
        pts2 = np.float32([[0, 0], [taille_grille, 0], [0, taille_grille], [
                          taille_grille, taille_grille]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        grille = cv2.warpPerspective(frame, M, (taille_grille, taille_grille))
        grille = cv2.cvtColor(grille, cv2.COLOR_BGR2GRAY)
        grille = cv2.adaptiveThreshold(
            grille, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 3)

        cv2.imshow("grille", grille)
        if flag == 0:

            grille_txt = []
            for y in range(9):
                ligne = ""
                for x in range(9):
                    y2min = y * case + marge
                    y2max = (y + 1) * case - marge
                    x2min = x * case + marge
                    x2max = (x + 1) * case - marge
                    cv2.imwrite("mat" + str(y) + str(x) + ".png",
                                grille[y2min:y2max, x2min:x2max])
                    img = grille[y2min:y2max, x2min:x2max]
                    x = img.reshape(1, 28, 28, 1)
                    if x.sum() > 10000:
                        prediction = classifier.predict_classes(x)
                        ligne += "{:d}".format(prediction[0])
                    else:
                        ligne += "{:d}".format(0)
                grille_txt.append(ligne)
            print(grille_txt)
            result = sol.sudoku(grille_txt)
        print("Resultat:", result)

        if result is not None:
            flag = 1
            fond = np.zeros(
                shape=(taille_grille, taille_grille, 3), dtype=np.float32)
            for y in range(len(result)):
                for x in range(len(result[y])):
                    if grille_txt[y][x] == "0":
                        cv2.putText(fond, "{:d}".format(result[y][x]), ((
                            x) * case + marge + 3, (y + 1) * case - marge - 3), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.9, (0, 0, 255), 1)
            M = cv2.getPerspectiveTransform(pts2, pts1)
            h, w, c = frame.shape
            fondP = cv2.warpPerspective(fond, M, (w, h))
            img2gray = cv2.cvtColor(fondP, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask = mask.astype('uint8')
            mask_inv = cv2.bitwise_not(mask)
            img1_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            img2_fg = cv2.bitwise_and(fondP, fondP, mask=mask).astype('uint8')
            dst = cv2.add(img1_bg, img2_fg)
            dst = cv2.resize(dst, (1080, 620))
            cv2.imshow("frame", dst)
            out.write(dst)

        else:
            frame = cv2.resize(frame, (1080, 620))
            cv2.imshow("frame", frame)
            out.write(frame)

    else:
        flag = 0
        frame = cv2.resize(frame, (1080, 620))
        cv2.imshow("frame", frame)
        out.write(frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


out.release()
cap.release()
cv2.destroyAllWindows()
