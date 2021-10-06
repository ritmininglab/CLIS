import numpy as np
from graphics import GraphWin,Image, Point
import matplotlib.pyplot as plt
from imageio import imread as imread
from imageio import imwrite as imsave
from skimage.transform import resize as imresize
import tkinter as tk
from tkinter import simpledialog


def collectLabel(nclicks, classlist):
    ROOT = tk.Tk()
    ROOT.withdraw()
    USER_INP = simpledialog.askstring(title="Annotation Label",
                                      prompt="Please type in class label:")
    strs = USER_INP.split(',')
    for i in range(nclicks):
        classlist.append(int(strs[i]))
    return classlist

def imshow2(datas, uncs, dimcls, h1, w1):
    filename0 = 'auxiliary/00.png'
    
    plt.figure()
    plt.imshow(datas, vmin=0, vmax=dimcls)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(filename0, transparent=True, bbox_inches="tight", pad_inches=0)
    
    img0 = imread(filename0)
    img0 = imresize(img0, (h1,w1))
    
    plt.figure()
    plt.imshow(uncs, cmap='Reds', vmin=0, vmax=1.6)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(filename0, transparent=True, bbox_inches="tight", pad_inches=0)
    
    img1 = imread(filename0)
    img1 = imresize(img1, (h1,w1))
    
    img = np.zeros((h1*2, w1,4))
    img[0:h1,:] = img0
    img[h1:,:] = img1
    imsave(filename0, img)

def clickXY(h1, w1, nclicks, clicks):
    filename = 'auxiliary/00.png'
    
    win = GraphWin("Annotation Click", w1, h1)
    myImage = Image(Point(int(w1/2),int(h1/2)), filename)
    myImage.draw(win)

    for i in range(nclicks):
        p = win.getMouse()
        clicks.append([int(p.getX()), int(p.getY())])
        print("clicked at:", p.getX(), p.getY())
    win.close()
    
    return clicks
