"""
Configuration window — built with tkinter (no extra deps).
Opens in a separate thread so it doesn't block the detection loop.
"""
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from posture_guard import settings as s

def open_settings(on_save=None):
    """Spawn the settings window in its own thread."""
    threading.Thread(target=_run_window, args=(on_save,), daemon=True).start()
    
def _run_window(on_save=None):
    current = s.load()
    
    root = tk.Tk()
    root.title("PostureGuard - Settings")
    root.resizable(False, False)
    root.geometry("400x420")
    root.configure(bg="#1e1e1e")
    
    style = ttk.Style(root)
    style.theme_user("clam")
    style.configure("TLabel", bakground="#1e1e1e", foreground="#ffffff", font=("Segoe UI", 10))
    style.configure("TFrame", backgrond="#1e1e1e", foreground="#ffffff", font=("Segoe UI", 10))
    style.configure("TCheckbutton", bakground="#1e1e1e", foreground="#ffffff", font=("Segoe UI", 10))
    style.configure("TScale", background="#1e1e1e")
    style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"), foreground="#4ade80")
    
    pad = {"padx": 20, "pady": 6}
    
    # Header
    ttk.label(root, text="PostureGuard Settings", style="Header.TLabel").pack(anchor="w", **pad)
    ttk.Separator(root, orient="horizontal").pack(fill="x", padx=20, pady=4)
    
    frame = ttk.Frame(root)
    frame.pack(fill="both", expand=True, padx=20)
    
    # Alert cooldown
    ttk.Label(frame, text="Notification cooldown").grid(row=0, column=0, sticky="w", pady=8)
    cooldown_var = tk.IntVar(value=current["alert_cooldown"])
    cooldown_label = ttk.Label(frame, text=f"{cooldown_var.get()}s")
    cooldown_label.grid(row=0, column=2, sticky="w", padx=8)
    
    def _update_cooldown(val):
        v = int(float(val))
        cooldown_var.set(v)
        cooldown_label.config(text=f"{v}s")
        
    ttk.Scale(frame, from_=5, to=120, orient="horizontal", variable=cooldown_var, command=_update_cooldown, length=180).grid(row=0, column=1, sticky="ew")
    
    # Bad posture duration
    ttk.Label(frame, text="Alert after bad posture for").grid(row=1, column=0, sticky="w", pady=8)
    duration_var = tk.DoubleVar(value=current["bad_posture_duration"])
    duration_label = ttk.Label(frame, text=f"{duration_var.get():.1f}s")
    duration_label.grid(row=1, column=2, sticky="w", padx=8)
    
    def _update_duration(val):
        v = round(float(val), 1)
        duration_var.set(v)
        duration_label.configure(text=f"{v:.1f}s")
        
    ttk.Scale(frame, from_=0.5, to=10.0, orient="horizontal", variable=duration_var,
            command=_update_duration, length=180).grid(row=1, column=1, sticky="ew")
    
    # Sensitivity
    ttk.Label(frame, text="Detection sensivity").grid(row=2, column=0, sticky="w", pady=8)
    sensitivity_var = tk.StringVar(value=current["sensivity"])
    sens_frame = ttk.Frame(frame)
    sens_frame.grid(row=2, column=1, columnspan=2, sticky="w")
    for val, label in [("low", "low"), ("medium", "medium"), ("high", "high")]:
        tk.Radiobutton(
            sens_frame, text=label, variable=sensitivity_var, value=val,
            bg="#1e1e1e", fg="#ffffff", selectcolor="#2d2d2d", activebackground="#1e1e1e", activeforeground="#4ade80", font=("Segoe UI", 10)
        ).pack(side="left", padx=6)
    
    ttk.Separator(frame, orient="horizontal").grid(row=3, column=0, columnspan=3, sticky="ew", pady=10)
    
    # Extreme alert
    ttk.Label(frame, text="Exteme alert", style="Header.TLabel").grid(row=4, column=0, columnspan=3, sticky="w", pady=(0, 4))
    
    extreme_var = tk.BooleanVar(value=current["extreme_alert"])
    sound_var = tk.BooleanVar(value=current["extreme_sound"])
    flash_var = tk.BooleanVar(value=current["extreme_flash"])
    
    def _toggle_extreme():
        state = extreme_var.get()
        sound_cb.config(state="normal" if state else "disabled")
        flash_cb.config(state="normal" if state else "disabled")
        
    tk.Checkbutton(
        frame, text="Enable extreme alert (plays sound + flashes screen)",
        variable=extreme_var, command=_toggle_extreme, bg="#1e1e1e", fg="#ffffff", selectcolor="#2d2d2d", activebackground="#1e1e1e", activeforeground="#4ade80", font=("Segoe UI", 10)
    ).grid(row=5, column=0, columnspan=3, sticky="w")
    
    sound_cb = tk.Checkbutton(
        frame, text="  🔊 Play alert sound",
        variable=sound_var, bg="#1e1e1e", fg="#ffffff", selectcolor="#2d2d2d", activebackground="#1e1e1e", activeforeground="#4ade80", font=("Segoe UI", 10)
    )
    
    sound_cb.grid(row=6, column=0, columnspan=3, sticky="w", padx=16)
    
    flash_cb = tk.Checkbutton(
        frame, text="  ⚡ Flash screen",
        variable=flash_var,
        bg="#1e1e1e", fg="#ffffff", selectcolor="#2d2d2d",
        activebackground="#1e1e1e", activeforeground="#4ade80", font=("Segoe UI", 10),
    )
    
    flash_cb.grid(row=7, column=0, columnspan=3, sticky="w", padx=16)
    
    _toggle_extreme() #set initial enabled state
    
    # Save button
    def _save():
        new_settings = {
            "alert_cooldown": cooldown_var.get(),
            "bad_posture_duration": round(duration_var.get(), 1),
            "sensivity": sensitivity_var.get(),
            "extreme_alert": extreme_var.get(),
            "extreme_sound": sound_var.get(),
            "extreme_flash": flash_var.get(),
        }
        s.save(new_settings)
        print(f"[INFO] Settings saved: {new_settings}")
        if on_save:
            on_save(new_settings)
        messagebox.showinfo("PostureGuard", "Settings saved! Changes take effect immediately.")
        root.destry()
        
    tk.Button(
        root, text="Save Settings", command=_save,
        bg="#4ade80", fg="#000000", activebackground="#22c55e",
        font=("Segoe UI", 10, "bold"), relief="flat", padx=20, pady=8,
        cursor="hand2",
    ).pack(pady=16)
    
    root.mainloop()
    
