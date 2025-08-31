import cv2
import numpy as np

# Índices de las webcams (ajusta según tu sistema)
WEBCAM_USB = 1

# Mapa de teclas para catalogar objetos
LABELS = {
    ord('1'): 'circulo',
    ord('2'): 'cuadrado',
    ord('3'): 'triangulo'
}

# Parámetros iniciales
WINDOW_NAME = "Deteccion en Vivo - Presiona 1/2/3 para Catalogar"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# Trackbars para ajustar umbral, tamaño de kernel y área mínima de contorno
cv2.createTrackbar("Threshold", WINDOW_NAME, 127, 255, lambda x: None)
cv2.createTrackbar("Kernel", WINDOW_NAME, 1, 20, lambda x: None)
cv2.createTrackbar("AreaMin", WINDOW_NAME, 1000, 20000, lambda x: None)


def main():
    cap = cv2.VideoCapture(WEBCAM_USB)
    if not cap.isOpened():
        print("Error al abrir la cámara")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Lectura de parámetros de trackbars
        th = cv2.getTrackbarPos("Threshold", WINDOW_NAME)
        k = cv2.getTrackbarPos("Kernel", WINDOW_NAME)
        k = k if k % 2 == 1 else k + 1
        min_area = cv2.getTrackbarPos("AreaMin", WINDOW_NAME)

        # Binarización y limpieza
        _, bw = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        clean = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

        # Encontrar contornos relevantes
        cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        relevant = [cnt for cnt in cnts if cv2.contourArea(cnt) >= min_area]

        # Dibujar contornos o rectángulos en verde
        annotated = frame.copy()
        for cnt in relevant:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Instrucciones de catalogación en la ventana
        instruction = "Catalogar: 1.circulo ; 2.cuadrado ; 3.triangulo"
        cv2.putText(annotated, instruction, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Mostrar resultado en vivo
        cv2.imshow(WINDOW_NAME, annotated)

        # Capturar teclas
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC para salir
            break

        # Si se presionó una de las teclas de catalogación
        if key in LABELS and relevant:
            label = LABELS[key]
            for idx, cnt in enumerate(relevant, start=1):
                moments = cv2.moments(cnt)
                hu = cv2.HuMoments(moments).flatten()
                print(f"Objeto {idx} etiquetado como: {label}")
                print("Invariantes de Hu:")
                for i, hval in enumerate(hu, start=1):
                    print(f"  Hu[{i}]: {hval}")
                print("---")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()