"""Simple training GUI for SilenceAI

Features:
- Select data folder, model output file, checkpoint folder
- Set epochs and batch size
- Start / Stop training (runs `python -m training.train` as a subprocess)
- Live stdout/stderr log display
- Save/load last settings
- Open output folder in Explorer

Usage:
  python -m gui.train_gui
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading, subprocess, sys, os, json, queue, time, shlex

CONFIG_FILE = os.path.join(os.path.dirname(__file__), '..', '.train_gui_config.json')

class TrainGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SilenceAI - Training GUI")
        self.geometry("820x600")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.proc = None
        self.log_q = queue.Queue()
        self._polling = False

        frm = ttk.Frame(self)
        frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # Data folder
        r = 0
        ttk.Label(frm, text="Data folder (processed):").grid(row=r, column=0, sticky=tk.W)
        self.data_entry = ttk.Entry(frm, width=60)
        self.data_entry.grid(row=r, column=1, sticky=tk.W, padx=6)
        ttk.Button(frm, text="Browse", command=self.browse_data).grid(row=r, column=2)

        # Model file
        r += 1
        ttk.Label(frm, text="Model output (file .h5/.tf):").grid(row=r, column=0, sticky=tk.W)
        self.model_entry = ttk.Entry(frm, width=60)
        self.model_entry.grid(row=r, column=1, sticky=tk.W, padx=6)
        ttk.Button(frm, text="Browse", command=self.browse_model).grid(row=r, column=2)

        # Checkpoint folder
        r += 1
        ttk.Label(frm, text="Checkpoint folder:").grid(row=r, column=0, sticky=tk.W)
        self.ckpt_entry = ttk.Entry(frm, width=60)
        self.ckpt_entry.grid(row=r, column=1, sticky=tk.W, padx=6)
        ttk.Button(frm, text="Browse", command=self.browse_ckpt).grid(row=r, column=2)

        # Hyperparameters
        r += 1
        ttk.Label(frm, text="Epochs:").grid(row=r, column=0, sticky=tk.W)
        self.epochs_var = tk.IntVar(value=30)
        ttk.Spinbox(frm, from_=1, to=1000, textvariable=self.epochs_var, width=8).grid(row=r, column=1, sticky=tk.W)
        ttk.Label(frm, text="Batch size:").grid(row=r, column=1, sticky=tk.E, padx=(120,0))
        self.batch_var = tk.IntVar(value=32)
        ttk.Spinbox(frm, from_=1, to=1024, textvariable=self.batch_var, width=8).grid(row=r, column=1, sticky=tk.E, padx=(200,0))

        # Extra args
        r += 1
        ttk.Label(frm, text="Extra CLI args:").grid(row=r, column=0, sticky=tk.W)
        self.extra_entry = ttk.Entry(frm, width=60)
        self.extra_entry.grid(row=r, column=1, sticky=tk.W, padx=6)

        # Buttons
        r += 1
        btn_frame = ttk.Frame(frm)
        btn_frame.grid(row=r, column=0, columnspan=3, pady=(8,12))
        self.start_btn = ttk.Button(btn_frame, text="Start Training", command=self.start_training)
        self.start_btn.pack(side=tk.LEFT, padx=6)
        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Open model folder", command=self.open_model_folder).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Save settings", command=self.save_settings).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Clear logs", command=self.clear_logs).pack(side=tk.LEFT, padx=6)

        # Log area
        log_frame = ttk.Frame(frm)
        log_frame.grid(row=r+1, column=0, columnspan=3, sticky=tk.NSEW)
        frm.rowconfigure(r+1, weight=1)
        frm.columnconfigure(1, weight=1)

        self.log_text = tk.Text(log_frame, wrap=tk.NONE, state=tk.NORMAL)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text['yscrollcommand'] = vsb.set

        # Load last settings
        self.load_settings()

        # poll logs
        self.after(200, self.poll_log_queue)

    def browse_data(self):
        d = filedialog.askdirectory(title="Select processed data folder")
        if d:
            self.data_entry.delete(0, tk.END); self.data_entry.insert(0, d)

    def browse_model(self):
        p = filedialog.asksaveasfilename(title="Select model output path", defaultextension='.h5', filetypes=[('HDF5','*.h5'),('TensorFlow','*.tf'),('All','*.*')])
        if p:
            self.model_entry.delete(0, tk.END); self.model_entry.insert(0, p)

    def browse_ckpt(self):
        d = filedialog.askdirectory(title="Select checkpoint folder")
        if d:
            self.ckpt_entry.delete(0, tk.END); self.ckpt_entry.insert(0, d)

    def start_training(self):
        data = self.data_entry.get().strip()
        model = self.model_entry.get().strip()
        ckpt = self.ckpt_entry.get().strip()
        epochs = int(self.epochs_var.get())
        batch = int(self.batch_var.get())
        extra = self.extra_entry.get().strip()

        if not data or not os.path.isdir(data):
            messagebox.showerror("Error", "Please select a valid processed data folder")
            return
        if not model:
            messagebox.showerror("Error", "Please select a model output file path")
            return
        # ensure output folder exists
        os.makedirs(os.path.dirname(model), exist_ok=True)
        if ckpt:
            os.makedirs(ckpt, exist_ok=True)

        cmd = [sys.executable, "-m", "training.train", "--data", data, "--model", model, "--epochs", str(epochs), "--batch-size", str(batch)]
        if ckpt:
            cmd += ["--checkpoint", ckpt]
        if extra:
            # naive split of extra args
            cmd += shlex.split(extra)

        self.log_text.insert(tk.END, f"Starting training: {' '.join(cmd)}\n")
        self.log_text.see(tk.END)
        self.start_btn['state'] = tk.DISABLED
        self.stop_btn['state'] = tk.NORMAL
        self.save_settings()

        # spawn subprocess in background thread
        def run_proc():
            try:
                self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)
                for line in self.proc.stdout:
                    self.log_q.put(line)
                self.proc.wait()
                rc = self.proc.returncode
                self.log_q.put(f"\nProcess exited with code {rc}\n")
            except Exception as e:
                self.log_q.put(f"Error launching training: {e}\n")
            finally:
                self.proc = None
                self.log_q.put("__TRAINING_DONE__")

        t = threading.Thread(target=run_proc, daemon=True)
        t.start()

    def stop_training(self):
        if self.proc:
            try:
                self.log_text.insert(tk.END, "Terminating training process...\n")
                self.log_text.see(tk.END)
                self.proc.terminate()
                # wait a bit then kill if still alive
                def kill_wait():
                    time.sleep(3)
                    if self.proc and self.proc.poll() is None:
                        try:
                            self.proc.kill()
                        except Exception:
                            pass
                threading.Thread(target=kill_wait, daemon=True).start()
            except Exception as e:
                self.log_q.put(f"Error stopping process: {e}\n")

    def open_model_folder(self):
        model = self.model_entry.get().strip()
        if not model:
            messagebox.showinfo("Info", "No model path set")
            return
        folder = os.path.dirname(model)
        if not os.path.isdir(folder):
            messagebox.showerror("Error", f"Folder does not exist: {folder}")
            return
        try:
            os.startfile(folder)
        except Exception:
            messagebox.showinfo("Info", f"Open folder: {folder}")

    def save_settings(self):
        cfg = {
            'data': self.data_entry.get().strip(),
            'model': self.model_entry.get().strip(),
            'ckpt': self.ckpt_entry.get().strip(),
            'epochs': int(self.epochs_var.get()),
            'batch': int(self.batch_var.get()),
            'extra': self.extra_entry.get().strip()
        }
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=2)
            self.log_q.put("Settings saved.\n")
        except Exception as e:
            self.log_q.put(f"Failed to save settings: {e}\n")

    def load_settings(self):
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                self.data_entry.delete(0, tk.END); self.data_entry.insert(0, cfg.get('data',''))
                self.model_entry.delete(0, tk.END); self.model_entry.insert(0, cfg.get('model',''))
                self.ckpt_entry.delete(0, tk.END); self.ckpt_entry.insert(0, cfg.get('ckpt',''))
                self.epochs_var.set(cfg.get('epochs', 30))
                self.batch_var.set(cfg.get('batch', 32))
                self.extra_entry.delete(0, tk.END); self.extra_entry.insert(0, cfg.get('extra',''))
        except Exception:
            pass

    def clear_logs(self):
        self.log_text.delete('1.0', tk.END)

    def poll_log_queue(self):
        try:
            while True:
                line = self.log_q.get_nowait()
                if line == '__TRAINING_DONE__':
                    self.start_btn['state'] = tk.NORMAL
                    self.stop_btn['state'] = tk.DISABLED
                else:
                    self.log_text.insert(tk.END, line)
                    self.log_text.see(tk.END)
        except queue.Empty:
            pass
        self.after(200, self.poll_log_queue)

    def on_close(self):
        if self.proc and self.proc.poll() is None:
            if not messagebox.askyesno("Exit", "Training is running. Terminate and exit?"):
                return
            self.stop_training()
            time.sleep(0.2)
        self.destroy()

if __name__ == '__main__':
    app = TrainGUI()
    app.mainloop()
