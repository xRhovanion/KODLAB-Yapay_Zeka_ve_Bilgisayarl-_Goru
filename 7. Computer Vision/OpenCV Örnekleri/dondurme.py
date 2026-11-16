import cv2

gorsel=cv2.imread("ornek3.png")

yukseklik, genislik = gorsel.shape[:2]   #Görüntüyü merkezin etrafında döndürmek için ikiye bölün.

dondurmeMatrisi = cv2.getRotationMatrix2D((genislik/2, yukseklik/2), -60,0.5)



gorsel = cv2.warpAffine(gorsel, dondurmeMatrisi, (genislik, yukseklik))


cv2.imwrite("dondurulmus.png",gorsel)
