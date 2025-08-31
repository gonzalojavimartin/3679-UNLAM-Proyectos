import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk

WEBCAM_MACBOOK = 0
WEBCAM_USB = 1

# ------------- 1. Carga de contornos de referencia (paso de clasificación) -------------
# Se leen imágenes limpias de cada forma (círculo, cuadrado, triángulo)
# y se extrae su contorno más grande para usarlo como modelo.
def load_reference_contours(ref_dir):
    ref_contours = {}
    for fname in os.listdir(ref_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        path = os.path.join(ref_dir, fname)
        name = os.path.splitext(fname)[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Binarizar imagen de referencia (paso 2 del enunciado: threshold)
        _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        # Extraer contornos (paso 4: findContours)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            # Guardar el contorno de mayor área como modelo
            ref_contours[name] = max(cnts, key=cv2.contourArea)
    return ref_contours

# ------------- 6. Función de comparación de formas (paso de clasificación) -------------
def classify_contour(cnt, ref_contours, tol):
    # Para cada contorno detectado, comparar con todos los modelos usando matchShapes()
    min_dist = float('inf')
    best = "Desconocido"
    for name, ref_cnt in ref_contours.items():
        dist = cv2.matchShapes(cnt, ref_cnt, cv2.CONTOURS_MATCH_I1, 0.0)
        if dist < min_dist:
            min_dist = dist
            best = name
    # Si la distancia al modelo más cercano es <= tolerancia, es reconocido; si no, desconocido
    if min_dist <= tol:
        return best, (0, 255, 0)  # verde para reconocidos (paso Output)
    return "Desconocido", (0, 0, 255)  # rojo para no reconocidos

def main():
    # Cargar contornos de referencia
    ref_contours = load_reference_contours("refs")

    # Defino el idex de la camara a utilizar
    webcam = cv2.VideoCapture(WEBCAM_USB)

    # Ventana única donde mostraremos Binaria + Resultado
    winform_name = "Deteccion de Formas"
    cv2.namedWindow(winform_name, cv2.WINDOW_NORMAL)
    # Trackbars para ajustar en tiempo real: threshold (paso 2), kernel morfológico (paso 3), area de tolerancia (paso 6)
    threshold_input = "Threshold"
    cv2.createTrackbar(threshold_input, winform_name, 127, 255, lambda x: None)

    kernel_input = "Kernel"
    cv2.createTrackbar(kernel_input, winform_name, 1, 20, lambda x: None)

    area_input = "Area"
    cv2.createTrackbar(area_input, winform_name, 10, 100, lambda x: None)

    while True:
        # Lectura de frame desde webcam (paso 0)
        ret, frame = webcam.read()
        if not ret:
            break

        # 2) Convertir a monocromático (escala de grises)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 3) Threshold binario con valor ajustable
        thresh = cv2.getTrackbarPos(threshold_input, winform_name)
        _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)

        # 4) Operación morfológica para eliminar ruido
        k = cv2.getTrackbarPos(kernel_input, winform_name)
        k = k if k % 2 == 1 else k + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        clean = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

        # 5) Detección de contornos en imagen limpia
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 6) Clasificar cada contorno con matchShapes()
        tol = cv2.getTrackbarPos(area_input, winform_name) / 100.0
        for cnt in contours:
            # Filtrar contornos espurios por área mínima
            if cv2.contourArea(cnt) < 500:
                continue
            # Clasificación y anotación (paso 7 Output)
            label, color = classify_contour(cnt, ref_contours, tol)
            x, y, _, _ = cv2.boundingRect(cnt)
            cv2.drawContours(frame, [cnt], -1, color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 7) Combinar y mostrar en un único frame: Binaria | Resultado
        h, w = frame.shape[:2]
        bw_bgr = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        bw_bgr = cv2.resize(bw_bgr, (w, h))
        combined = np.hstack((bw_bgr, frame))

        # Separador visual y etiquetas
        sep = (100, 100, 100)
        cv2.line(combined, (w, 0), (w, h), sep, 2)
        cv2.putText(combined, "Binaria", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, sep, 2)
        cv2.putText(combined, "Resultado", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, sep, 2)
        cv2.imshow(winform_name, combined)

        # Salir con tecla ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
