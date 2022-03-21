#!/usr/bin/env python
import tkinter as tk
import sys, os
from tkinter.constants import TOP
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter import Label
from PIL import Image
from logger import MyHandlerText
import os
import logging
import datetime

module_logger = logging.getLogger(__name__)

class application:

    def __init__(self, module_logger):
       self.window = tk.Tk()
       self.im = tk.PhotoImage(file="franka.png")
       self.logo = tk.PhotoImage(file="logo.png")
       self.label = tk.Label(image=self.im)
       self.head = tk.Label(self.window, text="      Simulator Manager", compound = tk.LEFT, relief="solid", background = "#8B919B", font=('Helvetica', 30), image=self.logo)
       self.txt_edit = tk.Text(self.window, height=10, width=120, background = "#D4CED3")
       self.logger = tk.Text(self.window, height=10, width=120)
       self.fr_buttons = tk.Frame(self.window, relief=tk.RAISED, bd=2)
       self.btn_open = tk.Button(self.fr_buttons, text="Open Simulation""\n""Configuration", height= 3, width=30, font=('Helvetica', 10), background = "#D4CED3", command=open_file)
       self.btn_save = tk.Button(self.fr_buttons, text="Save Simulation""\n""Configuration As...", height= 3, width=30, font=('Helvetica', 10), background = "#D4CED3", command=save_file)
       self.btn_start = tk.Button(self.fr_buttons, text="Select Simulation", height= 3, width=30, font=('Helvetica', 10), background = "#D4CED3", command=select_simulation)
       self.module_logger = module_logger
       
    def set_layout(self):
       self.window.title("Franka Emika - Simulator Manager")
       self.window.rowconfigure(0, minsize=100, weight=1)
       self.window.columnconfigure(1, minsize=100, weight=1) 
       self.btn_open.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
       self.btn_save.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
       self.btn_start.grid(row=2, column=0, sticky="ew", padx=10,pady=5)
       self.fr_buttons.grid(row=1, column=0, sticky="ns", pady=5)
       self.txt_edit.grid(row=1, column=1, sticky="nsew", pady=5)
       self.logger.grid(row=2, column=1, sticky="nsew") 
       self.head.grid(row=0, column=1, sticky="nsew",padx=(10, 10), pady=(7.5, 0), columnspan=2, rowspan=1)       
       self.label.grid(row=0, column=0, padx=10) 
       
    def link_handler(self, MyHandlerText):
       self.guiHandler = MyHandlerText(self.logger)
       self.module_logger.addHandler(self.guiHandler)
       self.module_logger.setLevel(logging.INFO)        


def open_file():
    """Open a file for editing."""
    filepath = askopenfilename(
        filetypes=[("All Files", "*.*")]
    )
    if not filepath:
        return
    app.txt_edit.delete(1.0, tk.END)
    with open(filepath, "r") as input_file:
        text = input_file.read()        
        app.txt_edit.insert(tk.END, text)
    app.window.title(f"Franka Emika - Simulator Manager - {filepath}")
    return

def save_file():
    """Save the current file as a new file."""
    filepath = asksaveasfilename(
        defaultextension="txt",
        filetypes=[("All Files", "*.*")],
    )
    if not filepath:
        return
    with open(filepath, "w") as output_file:
        text = app.txt_edit.get(1.0, tk.END)        
        output_file.write(text)
    app.window.title(f"Franka Emika - Simulator Manager - {filepath}")
    return

def select_simulation():
    filepath = askopenfilename(
        filetypes=[("All Files", "*.*")]
    ) 
    app.module_logger.info("Started Simulation: " + filepath.rsplit('/',1)[1])    
    os.system('python3 ' + filepath)    
    app.module_logger.info("Stopped Simulation: " + filepath.rsplit('/',1)[1]) 
    return 

#Launch
app = application(module_logger)     
app.set_layout()
app.link_handler(MyHandlerText)  
app.window.mainloop()

