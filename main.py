from dataclasses import dataclass
from enum import Enum
from sys import argv

import cv2
import easyocr
import langdetect
import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import MatLike
from langdetect import LangDetectException


class Language(Enum):
    """available languages enum"""

    SPANISH = "es"
    ENGLISH = "en"

    def __str__(self):
        """parse enum to its identifier in lowercase"""

        return self.name.lower()


# map supported languages to their color identifier
LanguageColor = {
    Language.SPANISH: (0, 255, 0),
    Language.ENGLISH: (255, 0, 0),
    None: (0, 0, 255),
}


#
class RecognitionResult:
    """result state after text recognition"""

    text: str
    language: Language | None
    image_processing: list[tuple[MatLike, str, dict]]  # (image, title, kwargs)


def recognize_language(text: str) -> tuple[Language | None, str]:
    """recognize the language in the text"""

    if len(text) < 10:
        return None, "text too short"

    # try to parse the language code
    try:
        lang_code = langdetect.detect(text)
        return Language(lang_code), "recognized"
    except ValueError:  # code not in Language enum
        return None, "language not supported"
    except LangDetectException as e:  # no language recognized
        if e.code == langdetect.lang_detect_exception.ErrorCode.CantDetectError:
            return None, "can't recognize language"
        return None, f"error: {str(e)}"


def simple_text_recognition(image_path: str) -> RecognitionResult:
    """simple text recognition, minimizing processing"""

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"invalid image path: '{image_path}'")

    # gray scale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # OTSU threshold binization (simpler than adaptive, Otsu finds optimal)
    _, binary = cv2.threshold(gray, 50, 250, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # filter noise with median blur (increment effect with greater kernel)
    denoised = cv2.medianBlur(binary, 3)

    # upgrade contrast with simple morphologic operations
    kernel = np.ones((2, 2), np.uint8)

    # dilate for connecting text elements
    dilated = cv2.dilate(denoised, kernel, iterations=1)

    # apply ocr in the original image
    recognized_text = READER.readtext(image)

    # parse fragments and bounding boxes
    text_fragments = []
    bounding_boxes = []
    for bbox, text, prob in recognized_text:
        if text.strip():
            text_fragments.append(text.strip())
            bounding_boxes.append((bbox, text, prob))

    complete_text = " ".join(text_fragments)

    # recognize language
    language, lang_recognition_st = recognize_language(complete_text)
    if language is None:
        print(lang_recognition_st)
    print(f"{language} {lang_recognition_st}")

    lang_color = LanguageColor[language]

    # draw bounding boxes in the original image
    bounded_image = image.copy()
    for bbox, text, prob in bounding_boxes:
        pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
        cv2.polylines(bounded_image, [pts], True, lang_color, 2)

        x, y = bbox[0]
        cv2.putText(
            bounded_image,
            f"{text}",
            (int(x), int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            lang_color,
            2,
        )

    return RecognitionResult(
        complete_text,
        language,
        [
            (image, "original", {}),
            (gray, "gray", {"cmap": "gray"}),
            (binary, "binary", {"cmap": "gray"}),
            (denoised, "denoised", {"cmap": "gray"}),
            (dilated, "dilated", {"cmap": "gray"}),
            (bounded_image, "bounded image", {}),
        ],
    )


def display_results(result: RecognitionResult) -> None:
    """displays resume of results"""

    print("\n===== RESULTS =====")
    print(f"language recognized: {result.language}")
    print(f"text recognized:\n{result.text}")

    # image processing visualization
    plt.figure(figsize=(15, 10))
    for image, title, kwargs in result.image_processing:
        plt.imshow(image, **kwargs)
        plt.title(title)

        plt.axis("off")
        plt.draw()

        plt.waitforbuttonpress()


def main() -> None:
    global READER

    if len(argv) != 2:
        print(f"Usage: {argv[0]} image_path")
        return

    # initialize easyocr reader
    READER = easyocr.Reader([lang.value for lang in Language])

    image_path = argv[1]
    result = simple_text_recognition(image_path)
    display_results(result)


if __name__ == "__main__":
    main()
