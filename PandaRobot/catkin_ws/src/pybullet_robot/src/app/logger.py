#!/usr/bin/env python
import tkinter as tk
import logging
import datetime

class MyHandlerText(logging.Handler):

	def __init__(self, widget):
		logging.Handler.__init__(self)
		self.setLevel(logging.NOTSET)		
		self.widget = widget
		self.widget.config(state='disabled')
		self.widget.tag_config("INFO", foreground ="black")
		self.widget.tag_config("DEBUG", foreground ="grey")
		self.widget.tag_config("WARNING", foreground ="orange")
		self.widget.tag_config("ERROR", foreground ="red")	      	
		self.widget.tag_config("CRITICAL", foreground ="red", underline=1)
		
		self.red = self.widget.tag_configure("red", foreground="red")
		
	def emit(self, record):
		self.widget.config(state='normal')		
		self.widget.insert(tk.END, datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S -> ") + self.format(record) + '\n')
		self.widget.see(tk.END)
		self.widget.config(state='disabled')
		self.widget.update()

		
		
