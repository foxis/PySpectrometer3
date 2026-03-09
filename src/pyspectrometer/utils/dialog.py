"""Simple dialog utilities."""


def prompt_calibrate() -> None:
    """Show dialog prompting user to run calibration when calibration data could not be loaded."""
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "Calibration Required",
            "Calibration data could not be loaded.\n\n"
            "Please run calibration mode first:\n\n"
            "  poetry run calibrate\n\n"
            "Then switch to measurement, Raman, or Color Science mode.",
        )
        root.destroy()
    except Exception:
        print("Calibration data could not be loaded. Run: poetry run calibrate")
