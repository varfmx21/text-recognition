# backnote-detector

Detector de texto en imagenes, usando librerias de python de procesamiento de
imagenes y OCR (optical character recognition).

Usamos opencv para realizar un flujo de filtros a la imagen, asi pudiendo
extraer la information textual de esta con easyocr, posteriormente mostramos el
proceso del flujo y el texto extraido a la terminal.

## Ejecucion

Instalar python 3.12, luego instalar las dependencias

```sh
virtualenv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

ejecutar con

```sh
source .venv/bin/activate
python3 main.py _path_to_image_
```

# Flujo de filtros

![processing flow img](URL "Image Processing Flow")

