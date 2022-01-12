# -*- coding: utf-8 -*-
# @Author: RangeKing
# @Original Author: yilin(https://github.com/insaneyilin/document_scanner)
# Python version: 3.8
# GUI reference: https://github.com/lancebeet/imagemicro

import os
import sys
import math
import argparse

from PIL import Image
from PIL import ImageTk

import tkinter
from tkinter import filedialog
import tkinter.messagebox
from pdf2image import convert_from_path
import numpy as np
import cv2

from note_shrink import (
    sample_pixels,
    get_palette,
    apply_palette
)


from doc_scanner import (
    get_document_corners,
    apply_four_point_perspective_transform
)

class ocrPreprocesWindow(object):
    def __init__(self, master, image_filename=None, pdf_filename=None):
        self.master = master
        self.origin_image = Image.new('RGB', (640,480), (255, 255, 255))
        self.image = Image.new('RGB', (640,480), (255, 255, 255))
        self.cv_image = None
        self.warped_image = None
        self.file_dir = ""
        self.filename = ""
        self.angle = 0
        self.init_window_size()
        self.init_menubar()

        # for select doc corners
        self.enable_select_corner = False
        self.selected_corner_idx = -1
        self.drawed_lines = [None]*4

        self.image_tmp = ImageTk.PhotoImage(self.image)

        # init four corners of the document
        self.canvas = tkinter.Canvas(self.master,
                width=self.master.winfo_screenwidth(),
                height=self.master.winfo_screenheight(),
                bd=0, highlightthickness=0)
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tkinter.NW,
                image=self.image_tmp)
        self.canvas.bind('<Motion>', self.on_mouse_move)
        self.canvas.pack()
        self.paint_ovals = [
                self.canvas.create_oval(0, 0, 0, 0, outline="#f11", fill="#1f1"),
                self.canvas.create_oval(0, 0, 0, 0, outline="#f11", fill="#1f1"),
                self.canvas.create_oval(0, 0, 0, 0, outline="#f11", fill="#1f1"),
                self.canvas.create_oval(0, 0, 0, 0, outline="#f11", fill="#1f1")]

        self.master.bind("<ButtonPress-1>", self.on_left_click)  
        # self.angle = tkinter.DoubleVar(self.master,value=90)
        # self.angle_scale = tkinter.Scale(self.master,variable = self.angle,
        #                                  from_ = -180, to = 180, 
        #                                  orient = tkinter.HORIZONTAL)
        # self.angle_buttonForget = tkinter.Button(self.master,
        #                   text = 'Hide',
        #                   command=lambda: [self.angle_scale.pack_forget(),self.angle_buttonForget.pack_forget()])      

        if image_filename is not None:
            self.open_file(image_filename)
        
        if pdf_filename is not None:
            self.open_pdf_file(pdf_filename)


    def init_window_size(self):
        self._geom='1080x720+0+0'
        pad = 0
        self.master.geometry("{0}x{0}+0+0".format(
                self.master.winfo_screenwidth() - pad,
                self.master.winfo_screenheight() - pad))
        # self.master.resizable(False, False)  # disable resizing


    def init_menubar(self):
        menubar = tkinter.Menu(self.master)
        file_menu = tkinter.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self.open_file)        
        file_menu.add_command(label="Save as", command=self.save_file)
        file_menu.add_command(label="Export to PDF", command=self.export2pdf)
        file_menu.add_command(label="Convert PDF to PNG", command=self.open_pdf_file)
        file_menu.add_command(label="Quit", command=self.master.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        tools_menu = tkinter.Menu(menubar, tearoff=0)    
        rotation_menu =  tkinter.Menu(menubar, tearoff=0)        
        tools_menu.add_cascade(label="Rotation", menu=rotation_menu)
        rotation_menu.add_command(label="Rotate 90° Clockwise", command=self.image_rotation_90_CLOCKWISE)
        rotation_menu.add_command(label="Rotate 90° Counterclockwise", command=self.image_rotation_90_COUNTERCLOCKWISE)
        tools_menu.add_command(label="Binarization", command=self.image_binarization)
        tools_menu.add_command(label="Erosion", command=self.image_erosion)
        tools_menu.add_command(label="Dilation", command=self.image_dilation)
        tools_menu.add_command(label="Invert Colors", command=self.invert_colors)
        tools_menu.add_command(label="Find Edges", command=self.edge_detect)
        tools_menu.add_command(label="Restore Image", command=self.restore_image)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        transform_menu = tkinter.Menu(menubar, tearoff=0)
        transform_menu.add_command(label="Detect Contour", command=self.detect_contour)
        transform_menu.add_command(label="Apply Perspective Transform",
                command=self.apply_perspective_transform)
        menubar.add_cascade(label="Document Scan", menu=transform_menu)    

        enhancing_menu = tkinter.Menu(menubar, tearoff=0)    
        enhancing_menu.add_command(label="Text Enhancement", command=self.image_enhancing)
        menubar.add_cascade(label="Enhancement", menu=enhancing_menu)

        self.master.config(menu=menubar)


    def open_file(self, filename=None):
        if filename is None:
            filename = filedialog.askopenfilename(parent=self.master,title='Choose image files', filetypes=[("image files", ".jpg .png")])
        if type(filename) is tuple or filename == "":
            return
        self.image = Image.open(filename).convert("RGB")
        self.origin_image = self.image.copy()
        print("filename: {}".format(filename))
        print("image size: [{} {}]".format(self.image.size[0], self.image.size[1]))

        self.file_dir, self.filename = os.path.split(filename)
        pad = 10
        image_w = self.image.size[0]
        image_h = self.image.size[1]
        self.doc_corners = [[pad, pad],
                [image_w - pad, pad],
                [image_w - pad, image_h - pad],
                [pad, image_h - pad]]
        self.update()

    def open_pdf_file(self, filename=None):
        if filename is None:
            filename = filedialog.askopenfilename(parent=self.master,title='Choose PDF files', filetypes=[("PDF files", ".pdf")])
        if type(filename) is tuple or filename == "":
            return
        print("filename: {}".format(filename))
        images = convert_from_path(filename)

        for i in range(len(images)):   
            # Save pages as images in the pdf
            images[i].save(f'./output/{os.path.basename(filename)[:-4]}_Page-{i+1}.png')
        tkinter.messagebox.showinfo('Conversion completed', f'Results saved in {os.path.dirname(os.path.realpath(__file__))}\output')

    def save_file(self):
        filename = filedialog.asksaveasfilename()
        if type(filename) is tuple or filename == "":
            return
        self.image.save(filename)
        tkinter.messagebox.showinfo('Save completed', f'Results saved as {filename}')


    def export2pdf(self):
        filename = filedialog.asksaveasfilename()
        if type(filename) is tuple or filename == "":
            return
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        self.image.save(filename)
        tkinter.messagebox.showinfo('Export completed', f'Results saved as {filename}')


    def update(self):
        self.master.geometry('{}x{}'.format(self.image.size[0], self.image.size[1]))

        self.image_tmp = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.image_on_canvas, image=self.image_tmp)

        for oval, corner in zip(self.paint_ovals, self.doc_corners):
            x, y = corner[0], corner[1]
            self.canvas.coords(oval, x-5, y-5, x+5, y+5)
        for idx, corner in enumerate(self.doc_corners):
            next_idx = (idx+1) % len(self.doc_corners)
            next_corner = self.doc_corners[next_idx]
            if self.drawed_lines[idx] is not None:
                self.canvas.delete(self.drawed_lines[idx])
            self.drawed_lines[idx] = self.canvas.create_line(corner[0], corner[1],
                    next_corner[0], next_corner[1],
                    dash=(4, 2), fill="#05f")

        self.master.wm_title(self.filename)


    def edge_detect(self):
        self.cv_image = cv2.cvtColor(np.asarray(self.image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 100)
        self.cv_image = cv2.cvtColor(edged, cv2.COLOR_GRAY2RGB)
        self.image = Image.fromarray(self.cv_image)
        self.update()


    def image_binarization(self):
        self.cv_image = cv2.cvtColor(np.asarray(self.image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.cv_image = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB)
        self.image = Image.fromarray(self.cv_image)
        self.update()
    
    def image_erosion(self):
        self.cv_image = cv2.cvtColor(np.asarray(self.image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), dtype=np.uint8)
        erosion = cv2.erode(gray, kernel, iterations=1)
        self.cv_image = cv2.cvtColor(erosion, cv2.COLOR_GRAY2RGB)
        self.image = Image.fromarray(self.cv_image)
        self.update()

    def image_dilation(self):
        self.cv_image = cv2.cvtColor(np.asarray(self.image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilation = cv2.dilate(gray, kernel, iterations=1)
        self.cv_image = cv2.cvtColor(dilation, cv2.COLOR_GRAY2RGB)
        self.image = Image.fromarray(self.cv_image)
        self.update()

    def invert_colors(self):
        inverted = 255 - np.asarray(self.image)
        self.image = Image.fromarray(inverted)
        self.update()

    ## TODO rotate any degree
    # def image_rotation(self):        
    #     self.angle_buttonForget.pack()
    #     self.angle_scale.pack()
    #     angle = float(self.angle.get())
    #     self.cv_image = cv2.cvtColor(np.asarray(self.image), cv2.COLOR_RGB2BGR)
    #     image_center = tuple(np.array(self.cv_image.shape[1::-1]) / 2)
    #     rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    #     rotation = cv2.warpAffine(self.cv_image, rot_mat, self.cv_image.shape[1::-1], flags=cv2.INTER_LINEAR)                
    #     self.cv_image = cv2.cvtColor(rotation, cv2.COLOR_BGR2RGB)
    #     self.image = Image.fromarray(self.cv_image)
    #     self.update()
    
    def image_rotation_90_CLOCKWISE(self):
        self.cv_image = cv2.cvtColor(np.asarray(self.image), cv2.COLOR_RGB2BGR)        
        rotation = cv2.rotate(self.cv_image, cv2.ROTATE_90_CLOCKWISE)                
        self.cv_image = cv2.cvtColor(rotation, cv2.COLOR_BGR2RGB)
        self.image = Image.fromarray(self.cv_image)
        self.update()

    def image_rotation_90_COUNTERCLOCKWISE(self):
        self.cv_image = cv2.cvtColor(np.asarray(self.image), cv2.COLOR_RGB2BGR)        
        rotation = cv2.rotate(self.cv_image, cv2.ROTATE_90_COUNTERCLOCKWISE)                
        self.cv_image = cv2.cvtColor(rotation, cv2.COLOR_BGR2RGB)
        self.image = Image.fromarray(self.cv_image)
        self.update()

    class options():
        def __init__(self):
            self.sample_fraction = 0.05
            self.quiet = False
            self.num_colors = 8
            self.value_threshold = 0.25
            self.sat_threshold = 0.20
            self.saturate = True

    def image_enhancing(self):
        options = self.options()
        self.image = np.asarray(self.image)
        samples = sample_pixels(self.image, options)
        palette = get_palette(samples, options)        
        labels = apply_palette(self.image, palette, options)
        self.image = Image.fromarray(labels, 'P')
        if options.saturate:
            palette = palette.astype(np.float32)
            pmin = palette.min()
            pmax = palette.max()
            palette = 255 * (palette - pmin)/(pmax-pmin)
            palette = palette.astype(np.uint8)
        self.image.putpalette(palette.flatten())
        self.update()


    def restore_image(self):
        self.image = self.origin_image.copy()
        self.update()


    def detect_contour(self):
        self.cv_image = cv2.cvtColor(np.asarray(self.image), cv2.COLOR_RGB2BGR)
        corners = get_document_corners(self.cv_image)
        if len(corners) == 4:
            self.doc_corners = [[pt[0], pt[1]] for pt in corners]
        self.update()


    def apply_perspective_transform(self):
        self.cv_image = cv2.cvtColor(np.asarray(self.image), cv2.COLOR_RGB2BGR)
        self.warped_image = apply_four_point_perspective_transform(self.cv_image,
                np.array(self.doc_corners))
        self.cv_image = cv2.cvtColor(self.warped_image, cv2.COLOR_BGR2RGB)
        self.image = Image.fromarray(self.cv_image)
        image_w = self.image.size[0]
        image_h = self.image.size[1]
        self.doc_corners = [[0, 0],
                [image_w, 0],
                [image_w, image_h],
                [0, image_h]]
        self.update()


    def on_mouse_move(self, event):
        x, y = event.x, event.y
        if self.enable_select_corner:
            self.doc_corners[self.selected_corner_idx] = [x, y]
            self.update()


    def on_left_click(self, event):
        # print("({}, {}) clicked".format(event.x, event.y))
        if self.enable_select_corner:
            self.enable_select_corner = False
            self.selected_corner_idx = -1
            return
        x, y = event.x, event.y
        for idx, corner in enumerate(self.doc_corners):
            if math.hypot(x-corner[0], y-corner[1]) < 10:
                self.selected_corner_idx = idx
                self.enable_select_corner = True
                break


def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--image', type=str, help='input image filename',
            default=None)
    return arg_parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    image_filename = args.image

    root = tkinter.Tk()
    root.title('Preprocess Tool for OCR')
    
    doc_scan_window = ocrPreprocesWindow(root, image_filename)
    
    root.mainloop()
