import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr

# Inicializar EasyOCR una sola vez (proceso costoso)
reader = easyocr.Reader(['es', 'en'])

def simple_text_detection(image_path):
    """Enfoque simplificado para detección de texto con preprocesamiento mínimo."""
    # 1. Cargar imagen
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: '{image_path}'")
    
    # 2. Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 3. Binarización con umbral de Otsu (más simple que adaptativa)
    # Otsu encuentra automáticamente el umbral óptimo
    _, binary = cv2.threshold(gray, 50, 250, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. Opcional: Eliminar ruido con filtro mediana
    denoised = cv2.medianBlur(binary, 1)
    
    # 5. Mejorar contraste con operaciones morfológicas simples
    # Dilatación para conectar componentes de texto
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(denoised, kernel, iterations=1)
    
    # 6. Aplicar OCR directamente sobre la imagen original
    # (EasyOCR suele funcionar bien con la imagen original)
    results = reader.readtext(image)
    
    # 7. Preparar visualización
    result_image = image.copy()
    text_found = []
    
    for (bbox, text, prob) in results:
        # Dibujar rectángulos alrededor del texto
        pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
        cv2.polylines(result_image, [pts], True, (0, 255, 0), 2)
        
        # Añadir texto a la lista de resultados
        if text.strip():
            text_found.append(text)
            print(f"Texto detectado ({prob:.2f}): {text}")
    
    # 8. Visualización del proceso
    plt.figure(figsize=(15, 10))
    plt.subplot(231), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original')
    plt.subplot(232), plt.imshow(gray, cmap='gray'), plt.title('Escala de grises')
    plt.subplot(233), plt.imshow(binary, cmap='gray'), plt.title('Binarización Otsu')
    plt.subplot(234), plt.imshow(denoised, cmap='gray'), plt.title('Eliminación de ruido')
    plt.subplot(235), plt.imshow(dilated, cmap='gray'), plt.title('Dilatación')
    plt.subplot(236), plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)), plt.title('Resultado')
    plt.tight_layout()
    plt.show()
    
    return text_found

if __name__ == "__main__":
    # Ejecutar la detección simplificada
    image_path = "img-02.jpg"
    detected_text = simple_text_detection(image_path)
    
    # Mostrar todos los textos encontrados
    print("\nResumen de texto extraído:")
    for i, text in enumerate(detected_text, 1):
        print(f"{i}. {text}")