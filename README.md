# Bitirme-Projesi-Yüz Tanıma Uygulaması

import cv2
import os
import numpy as np

# Yüz görüntülerini saklamak için dizini oluştur, yoksa
faces_dir = 'faces'
if not os.path.exists(faces_dir):
    os.makedirs(faces_dir)

# Yüz tespiti için önceden eğitilmiş Haar Cascade sınıflandırıcısını yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Web kamerasını başlat
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Hata: Video akışı açılamadı.")
    exit()

known_faces = []
capture_count = 0
capture_count_max = 30  # Aynı kişi olarak kabul edilen başlangıçtaki yüzlerin sayısı
scanning_mode = False
name = ""

while True:
    # Kareyi kare olarak yakala
    ret, frame = cap.read()

    if not ret:
        print("Hata: Kare okunamadı.")
        break

    # Yüz tespiti için kareyi gri tonlamaya dönüştür
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Karedeki yüzleri tespit et
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE)

    # Yüzlerin etrafına dikdörtgen çiz ve etiketle
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_id = len(known_faces)

        # Bu yüzün zaten bilinen bir yüz olup olmadığını kontrol et
        match_found = False
        for idx, (known_face, _) in enumerate(known_faces):
            known_gray = cv2.cvtColor(known_face, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(known_gray, gray[y:y+h, x:x+w], cv2.TM_CCOEFF_NORMED)
            if np.max(res) >= 0.75:  # Gerekirse eşik değerini ayarlayın
                match_found = True
                face_id = idx
                break

        if not match_found:

            if not scanning_mode and cv2.waitKey(1) & 0xFF == ord('b'):
                name = input("Yeni yüz için isim giriniz: ")
                scanning_mode = True

            elif scanning_mode and capture_count <= capture_count_max:
                face_path = os.path.join(faces_dir, f'{name}_{face_id}.png')
                cv2.imwrite(face_path, face_img)
                known_faces.append((face_img, name))
                print(len(known_faces))
                capture_count += 1
                if capture_count == capture_count_max:
                    print(f"{name} için tarama tamamlandı...")
                    scanning_mode = False
                    capture_count = 0

        # Dikdörtgen çiz ve etiketle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        label = f'{known_faces[face_id][1]}' if face_id < len(known_faces) else "Tanimsiz"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Sonuç karesini göster
    cv2.imshow('Yüz Tespiti', frame)

    # 'q' tuşuna basıldığında döngüyü kır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Yakalama bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
