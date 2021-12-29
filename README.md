# OCR_preprocessing_tool

A simple OCR preprocessing tool using Python with a GUI.</br>

This repo is modified from https://github.com/insaneyilin/document_scanner, and `note_shrink.py` is modified from https://github.com/mzucker/noteshrink.

---

## Usage

- GUI - image rotation, binarization, edge detection, dilation/erosion, automatic/manual doc scanner, and pdf to png conversion:
  ```
  python OCR_preprocessing_tool.py
  ```
  
- Command Line - automatic doc scanner:

  ```
  python doc_scanner_app.py --image=<input_image_path>
  ```

- Command Line - text compressing and enhancing:

  ```
  python note_shrink.py IMAGE <input_image_path>
  ```
  
  Run the code below for more tips:
  
  ```
  python note_shrink.py -h
  ```

---

## Dependencies

- Python 3
- Tkinter
- OpenCV
- Pillow
- NumPy
- Scipy
- pdf2image

```
pip install -r requirements.txt
```

---

## Demo

### Rotation

<div align=center><img src="https://github.com/RangeKing/OCR_preprocessing_tool/blob/main/demo/rotation.gif"></div>

### Binarization

<div align=center><img src="https://github.com/RangeKing/OCR_preprocessing_tool/blob/main/demo/binarization.gif"></div>

### Edge detection

<div align=center><img src="https://github.com/RangeKing/OCR_preprocessing_tool/blob/main/demo/find_edges.gif"></div>

### Erosion

<div align=center><img src="https://github.com/RangeKing/OCR_preprocessing_tool/blob/main/demo/erosion.gif"></div>

### Dilation

<div align=center><img src="https://github.com/RangeKing/OCR_preprocessing_tool/blob/main/demo/dilation.gif"></div>

### Select corners manually

<div align=center><img src="https://github.com/RangeKing/OCR_preprocessing_tool/blob/main/demo/select_corners_manually.gif"></div>

### Auto detection (not very robust)

<div align=center><img src="https://github.com/RangeKing/OCR_preprocessing_tool/blob/main/demo/auto_detection.gif"></div>

### Text enhancement (after applying perspective transform)

<div align=center><img src="https://github.com/RangeKing/OCR_preprocessing_tool/blob/main/demo/text_enhancement.gif"></div>

### Conversion of pdf to png 

<div align=center><img src="https://github.com/RangeKing/OCR_preprocessing_tool/blob/main/demo/pdf_to_png.gif"></div>

---

## References

https://github.com/insaneyilin/document_scanner

https://github.com/mzucker/noteshrink

http://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/

https://www.geeksforgeeks.org/convert-pdf-to-image-using-python/

https://www.geeksforgeeks.org/how-to-hide-recover-and-delete-tkinter-widgets/

http://vipulsharma20.blogspot.com/2016/01/document-scanner-using-python-opencv.html

https://github.com/lancebeet/imagemicro

