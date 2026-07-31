"""
Configuration window — built with tkinter (no extra deps).
Opens in a separate thread so it doesn't block the detection loop.
"""
import threading
import tkinter as tk
from tkinter import messagebox
from posture_guard import settings as s

def open_settings(on_save=None):
    """Spawn the settings window in its own thread."""
    threading.Thread(target=_run_window, args=(on_save,), daemon=True).start()
    
def _run_window(on_save=None):
    from posture_guard import __version__
    current = s.load()
    
    root = tk.Tk()
    root.title("PostureGuard - Settings")
    root.resizable(False, False)
    
    # ── Fonts & colors ─────────────────────────────────────────────────────────
    F      = ("Segoe UI", 10)
    F_SM   = ("Segoe UI", 9)
    F_BOLD = ("Segoe UI", 10, "bold")
    F_CAP  = ("Segoe UI", 8)
    
    BG       = "#18181b"
    CARD     = "#27272a"
    BORDER   = "#3f3f46"
    TEXT     = "#f4f4f5"
    MUTED    = "#a1a1aa"
    ACCENT   = "#3b82f6"
    ACCENT_B = "#1d4ed8"
    PILL_BG  = "#1e3a5f"
    PILL_FG  = "#93c5fd"
    WARN_BG  = "#451a03"
    WARN_FG  = "#fb923c"
    
    root.configure(bg=BG)
    
    def card_frame(parent, **kwargs):
        f = tk.Frame(parent, bg=CARD, bd=0, highlightthickness=1, **kwargs)
        return f
    
    def section_label(parent, text):
        tk.Label(parent, text=text.upper(), bg=BG, fg=MUTED, font=F_CAP, anchor="w").pack(fill="x", padx=20, pady=(14, 4))
        
    def row_frame(parent):
        f = tk.Frame(parent, bg=CARD)
        f.pack(fill="x", padx=14, pady=6)
        return f
    
    def divider(parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=14)
        
    # Window layout
    # Title bar
    title_bar = tk.Frame(root, bg=BG)
    title_bar.pack(fill="x", padx=20, pady=(16, 4))
    tk.Label(title_bar, text="Settings", bg=BG, fg=TEXT,
             font=("Segoe UI", 14, "bold"), anchor="w").pack(side="left")
    tk.Label(title_bar, text=f"v{__version__}", bg=BG, fg=MUTED,
             font=F_SM, anchor="e").pack(side="right", pady=4)
    
    tk.Frame(root, bg=BORDER, height=1).pack(fill="x", padx=20, pady=(4, 0))
    
    # Alerts section
    section_label(root, "Alerts")
    alerts_card = card_frame(root)
    alerts_card.pack(fill="x", padx=20)
    
    # Cooldown slider
    cooldown_row = row_frame(alerts_card)
    tk.Label(cooldown_row, text="Notification cooldown", bg=CARD, fg=TEXT, font=F,
             anchor="w").pack(side="left")
    cooldown_chip = tk.Label(cooldown_row, text=f"{current['alert_cooldown']}s",
                             bg=PILL_BG, fg=PILL_FG, font=F_SM, width=5,
                             relief="flat", padx=6, pady=2)
    cooldown_chip.pack(side="right")
    cooldown_var = tk.IntVar(value=current["alert_cooldown"])
    
    def _upd_cooldown(val):
        v = int(float(val))
        cooldown_var.set(v)
        cooldown_chip.config(text=f"{v}s")
        
    tk.Scale(cooldown_row, from_=5, to=120, orient="horizontal", variable=cooldown_var, command=_upd_cooldown, length=160, bg=CARD, fg=TEXT, troughcolor=BORDER, highlightthickness=0, activebackground=ACCENT, bd=0, showvalue=False, sliderrelief="flat").pack(side="right", padx=8)
    
    divider(alerts_card)
    
    # Duration slider
    duration_row = row_frame(alerts_card)
    tk.Label(duration_row, text="Alert after bad posture for", bg=CARD, fg=TEXT, font=F, anchor="w").pack(side="left")
    dur_val = round(current["bad_posture_duration"], 1)
    duration_chip = tk.Label(duration_row, text=f"{dur_val:.1f}s", bg=WARN_BG, fg=WARN_FG, font=F_SM, width=5, relief="flat", padx=6, pady=2)
    
    duration_chip.pack(side="right")
    duration_var = tk.DoubleVar(value=current["bad_posture_duration"])
    
    def _upd_duration(val):
        v = round(float(val), 1)
        duration_var.set(v)
        duration_chip.config(text=f"{v:.1f}s")
    
    tk.Scale(duration_row, from_=0.5, to=10.0, resolution=0.5, orient="horizontal",
        variable=duration_var, command=_upd_duration, length=160,
            bg=CARD, fg=TEXT, troughcolor=BORDER, highlightthickness=0,
            activebackground=ACCENT, bd=0, showvalue=False,
            sliderrelief="flat").pack(side="right", padx=8)
 
    # ── Sensitivity section ────────────────────────────────────────────────────
    section_label(root, "Detection sensitivity")
    sens_card = card_frame(root)
    sens_card.pack(fill="x", padx=20)
    
    sens_row = row_frame(sens_card)
    tk.Label(sens_row, text="Sensitivity", bg=CARD, fg=TEXT, font=F, anchor="w").pack(side="left")
    
    sensitivity_var = tk.StringVar(value=current.get("sensitivity", "medium"))
    pill_btns = {}
    
    def _select_sens(val):
        sensitivity_var.set(val)
        for v, btn in pill_btns.items():
            if v == val:
                btn.config(bg=ACCENT, fg="white")
            else:
                btn.config(bg=CARD, fg=MUTED)
    
    pill_frame = tk.Frame(sens_row, bg=CARD)
    pill_frame.pack(side="right")
    for val, label in [("low", "Low"), ("medium", "Medium"), ("high", "High")]:
        is_active = sensitivity_var.get() == val
        btn = tk.Button(
            pill_frame, text=label, font=F_SM,
            bg=ACCENT if is_active else BORDER,
            fg="white" if is_active else MUTED,
            activebackground=ACCENT_B, activeforeground="white",
            relief="flat", padx=10, pady=3, cursor="hand2",
            command=lambda v=val: _select_sens(v),
        )
        btn.pack(side="left", padx=2)
        pill_btns[val] = btn
 
    # ── Extreme alert section ──────────────────────────────────────────────────
    section_label(root, "Extreme alert")
    extreme_card = card_frame(root)
    extreme_card.pack(fill="x", padx=20)
 
    extreme_var = tk.BooleanVar(value=current.get("extreme_alert", False))
    sound_var   = tk.BooleanVar(value=current.get("extreme_sound", True))
    flash_var   = tk.BooleanVar(value=current.get("extreme_flash", True))
 
    def _toggle_sub():
        state = "normal" if extreme_var.get() else "disabled"
        fg_col = TEXT if extreme_var.get() else MUTED
        sound_cb.config(state=state, fg=fg_col)
        flash_cb.config(state=state, fg=fg_col)
 
    extreme_row = row_frame(extreme_card)
    tk.Checkbutton(
        extreme_row, text="Enable extreme alert", variable=extreme_var,
        command=_toggle_sub, bg=CARD, fg=TEXT, selectcolor=BORDER,
        activebackground=CARD, activeforeground=TEXT, font=F_BOLD,
        anchor="w", cursor="hand2",
    ).pack(side="left")
 
    divider(extreme_card)
 
    sub_frame = tk.Frame(extreme_card, bg=CARD)
    sub_frame.pack(fill="x", padx=28, pady=(4, 8))
 
    sound_cb = tk.Checkbutton(
        sub_frame, text="Play alert sound", variable=sound_var,
        bg=CARD, fg=TEXT if extreme_var.get() else MUTED,
        selectcolor=BORDER, activebackground=CARD, activeforeground=TEXT,
        font=F, anchor="w", cursor="hand2",
        state="normal" if extreme_var.get() else "disabled",
    )
    sound_cb.pack(anchor="w", pady=2)
 
    flash_cb = tk.Checkbutton(
        sub_frame, text="Flash screen", variable=flash_var,
        bg=CARD, fg=TEXT if extreme_var.get() else MUTED,
        selectcolor=BORDER, activebackground=CARD, activeforeground=TEXT,
        font=F, anchor="w", cursor="hand2",
        state="normal" if extreme_var.get() else "disabled",
    )
    flash_cb.pack(anchor="w", pady=2)
 
    # ── Save button ────────────────────────────────────────────────────────────
    tk.Frame(root, bg=BORDER, height=1).pack(fill="x", padx=20, pady=(14, 0))
 
    def _save():
        new_settings = {
            "alert_cooldown":       cooldown_var.get(),
            "bad_posture_duration": round(duration_var.get(), 1),
            "sensitivity":          sensitivity_var.get(),
            "extreme_alert":        extreme_var.get(),
            "extreme_sound":        sound_var.get(),
            "extreme_flash":        flash_var.get(),
        }
        s.save(new_settings)
        print(f"[INFO] Settings saved: {new_settings}")
        if on_save:
            on_save(new_settings)
        messagebox.showinfo("PostureGuard", "Settings saved.")
        root.destroy()
 
    tk.Button(
        root, text="Save settings", command=_save,
        bg=ACCENT, fg="white", activebackground=ACCENT_B, activeforeground="white",
        font=F_BOLD, relief="flat", padx=20, pady=10, cursor="hand2", bd=0,
    ).pack(fill="x", padx=20, pady=14)
 
    root.update_idletasks()
    root.minsize(root.winfo_width(), root.winfo_height())
    root.mainloop()    
