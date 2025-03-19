import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
from langdetect import detect, LangDetectException
from langdetect.lang_detect_exception import ErrorCode

# Inicializar EasyOCR una sola vez (proceso costoso)
reader = easyocr.Reader(['es', 'en'])

LANGUAGE_NAMES = {
    'es': 'Español',
    'en': 'Inglés',
}

def detect_language(text):
    """Detecta el idioma del texto proporcionado.
    
    Args:
        text: Texto a analizar
        
    Returns:
        Tuple: (código_idioma, nombre_idioma)
    """
    if not text or len(text) < 10:  # Aumentado el mínimo para mejor detección
        return "desconocido", "Texto demasiado corto"
    
    try:
        lang_code = detect(text)
        lang_name = LANGUAGE_NAMES.get(lang_code, f"Desconocido ({lang_code})")
        return lang_code, lang_name
    except LangDetectException as e:
        if e.code == ErrorCode.CantDetectError:
            return "desconocido", "No se pudo detectar"
        return "error", f"Error: {str(e)}"

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
    
    # 4. Opcional: Eliminar ruido con filtro mediana (aumentamos kernel para mayor efecto)
    denoised = cv2.medianBlur(binary, 3)
    
    # 5. Mejorar contraste con operaciones morfológicas simples
    # Dilatación para conectar componentes de texto
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(denoised, kernel, iterations=1)
    
    # 6. Aplicar OCR directamente sobre la imagen original
    # (EasyOCR suele funcionar bien con la imagen original)
    results = reader.readtext(image)
    
    # 7. Preparar visualización
    result_image = image.copy()
    
    # Recolectar todo el texto en un solo string
    all_text_fragments = []
    bounding_boxes = []
    
    for (bbox, text, prob) in results:
        if text.strip():
            all_text_fragments.append(text.strip())
            bounding_boxes.append((bbox, text, prob))
    
    # Unir todos los fragmentos de texto con espacios
    complete_text = " ".join(all_text_fragments)
    print(f"\nTexto completo extraído:\n{complete_text}\n")
    
    # Detectar idioma del texto completo
    lang_code, lang_name = detect_language(complete_text)
    print(f"Idioma detectado: {lang_name} ({lang_code})")
    
    # Color según idioma determinado
    if lang_code == 'es':
        color = (0, 255, 0)  # Verde para español
    elif lang_code == 'en':
        color = (255, 0, 0)  # Azul para inglés
    else:
        color = (0, 0, 255)  # Rojo para otros idiomas
    
    # Dibujar todos los rectángulos con el mismo color (del idioma detectado)
    for (bbox, text, prob) in bounding_boxes:
        pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
        cv2.polylines(result_image, [pts], True, color, 2)
        
        # Mostrar información en la consola para cada fragmento
        print(f"Fragmento detectado ({prob:.2f}): '{text}'")
        
        # Añadir etiqueta en la imagen
        x, y = bbox[0]
        cv2.putText(result_image, f"{text}", (int(x), int(y)-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 8. Visualización del proceso
    plt.figure(figsize=(15, 10))
    plt.subplot(231), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original')
    plt.subplot(232), plt.imshow(gray, cmap='gray'), plt.title('Escala de grises')
    plt.subplot(233), plt.imshow(binary, cmap='gray'), plt.title('Binarización Otsu')
    plt.subplot(234), plt.imshow(denoised, cmap='gray'), plt.title('Eliminación de ruido')
    plt.subplot(235), plt.imshow(dilated, cmap='gray'), plt.title('Dilatación')
    plt.subplot(236), plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)), plt.title(f'Resultado: {lang_name}')
    plt.tight_layout()
    plt.show()
    
    # Crear un diccionario con el texto completo y su idioma
    result = {
        'text': complete_text,
        'language_code': lang_code,
        'language': lang_name,
    }
    
    return result

def display_results(result):
    """Muestra un resumen de los resultados."""
    print("\n===== RESUMEN DE ANÁLISIS =====")
    print(f"Idioma detectado: {result['language']}")
    print(f"Texto completo:\n{result['text']}")

if __name__ == "__main__":
    # Ejecutar la detección simplificada
    image_path = "img-03.jpeg"
    result = simple_text_detection(image_path)
    
    # Mostrar resumen
    display_results(result)