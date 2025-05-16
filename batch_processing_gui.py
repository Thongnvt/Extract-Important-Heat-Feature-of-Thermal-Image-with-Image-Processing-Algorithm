import os
import sys
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading

class ThermalBatchProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Thermal Image Batch Processor")
        self.root.geometry("700x500")
        self.root.resizable(True, True)
        
        # Set up the main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Input folder selection
        input_frame = ttk.LabelFrame(main_frame, text="Input Folder", padding="10")
        input_frame.pack(fill=tk.X, pady=10)
        
        self.input_path = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.input_path, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(input_frame, text="Browse...", command=self.browse_input).pack(side=tk.RIGHT, padx=5)
        
        # Output folder selection
        output_frame = ttk.LabelFrame(main_frame, text="Output Folder", padding="10")
        output_frame.pack(fill=tk.X, pady=10)
        
        self.output_path = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.output_path, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="Browse...", command=self.browse_output).pack(side=tk.RIGHT, padx=5)
        
        # Enhancement method selection
        method_frame = ttk.LabelFrame(main_frame, text="Enhancement Method", padding="10")
        method_frame.pack(fill=tk.X, pady=10)
        
        self.method = tk.IntVar(value=1)
        methods = [
            (1, "CLAHE (Basic enhancement)"),
            (2, "Multi-Scale Guided Filtering (Recommended for general enhancement)"),
            (3, "Multiscale Top-Hat Transform (Good for highlighting both bright and dark regions)"),
            (4, "Region-Specific Alcohol Detection (Specialized for facial thermal features)"),
            (5, "Combined Approach (Best for alcohol detection)")
        ]
        
        for val, text in methods:
            ttk.Radiobutton(method_frame, text=text, variable=self.method, value=val).pack(anchor=tk.W, pady=2)
        
        # Color preservation option
        color_frame = ttk.Frame(main_frame, padding="10")
        color_frame.pack(fill=tk.X, pady=5)
        
        self.preserve_color = tk.BooleanVar(value=True)
        ttk.Checkbutton(color_frame, text="Preserve original thermal color spectrum", variable=self.preserve_color).pack(anchor=tk.W)
        
        # Process button and progress bar
        button_frame = ttk.Frame(main_frame, padding="10")
        button_frame.pack(fill=tk.X, pady=10)
        
        self.progress = ttk.Progressbar(button_frame, orient=tk.HORIZONTAL, length=300, mode='indeterminate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.process_button = ttk.Button(button_frame, text="Process Images", command=self.process_images)
        self.process_button.pack(side=tk.RIGHT, padx=5)
        
        # Status label
        self.status = tk.StringVar(value="Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status, font=("Arial", 10))
        status_label.pack(anchor=tk.W, pady=10)
        
        # Command preview
        preview_frame = ttk.LabelFrame(main_frame, text="Command Preview", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.command_preview = tk.Text(preview_frame, height=5, wrap=tk.WORD)
        self.command_preview.pack(fill=tk.BOTH, expand=True)
        
        # Update command preview when any option changes
        self.input_path.trace_add("write", self.update_command_preview)
        self.output_path.trace_add("write", self.update_command_preview)
        self.method.trace_add("write", self.update_command_preview)
        self.preserve_color.trace_add("write", self.update_command_preview)
        
        # Initial command preview update
        self.update_command_preview()
    
    def browse_input(self):
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            self.input_path.set(folder)
    
    def browse_output(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_path.set(folder)
    
    def update_command_preview(self, *args):
        input_path = self.input_path.get() or "<input_folder>"
        output_path = self.output_path.get() or "<output_folder>"
        method = self.method.get()
        color_flag = "--color" if self.preserve_color.get() else ""
        
        command = f"python batch_processing.py --input \"{input_path}\" --output \"{output_path}\" --method {method} {color_flag}"
        self.command_preview.delete(1.0, tk.END)
        self.command_preview.insert(tk.END, command)
    
    def process_images(self):
        # Validate inputs
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select an input folder")
            return
        
        if not self.output_path.get():
            messagebox.showerror("Error", "Please select an output folder")
            return
        
        if not os.path.exists(self.input_path.get()):
            messagebox.showerror("Error", f"Input folder does not exist: {self.input_path.get()}")
            return
        
        # Disable the process button and start progress bar
        self.process_button.config(state=tk.DISABLED)
        self.progress.start()
        self.status.set("Processing images...")
        
        # Get the command
        input_path = self.input_path.get()
        output_path = self.output_path.get()
        method = self.method.get()
        color_flag = "--color" if self.preserve_color.get() else ""
        
        command = f"python batch_processing.py --input \"{input_path}\" --output \"{output_path}\" --method {method} {color_flag}"
        
        # Run the command in a separate thread
        thread = threading.Thread(target=self.run_command, args=(command,))
        thread.daemon = True
        thread.start()
    
    def run_command(self, command):
        try:
            exit_code = os.system(command)
            
            # Update UI in the main thread
            self.root.after(0, self.process_complete, exit_code)
        except Exception as e:
            # Update UI in the main thread
            self.root.after(0, self.process_error, str(e))
    
    def process_complete(self, exit_code):
        self.progress.stop()
        self.process_button.config(state=tk.NORMAL)
        
        if exit_code == 0:
            self.status.set("Processing completed successfully")
            messagebox.showinfo("Success", f"All images processed successfully.\nEnhanced images saved to: {self.output_path.get()}")
        else:
            self.status.set(f"Processing failed with exit code {exit_code}")
            messagebox.showerror("Error", f"Processing failed with exit code {exit_code}")
    
    def process_error(self, error_message):
        self.progress.stop()
        self.process_button.config(state=tk.NORMAL)
        self.status.set(f"Error: {error_message}")
        messagebox.showerror("Error", f"An error occurred: {error_message}")


def main():
    root = tk.Tk()
    app = ThermalBatchProcessorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
