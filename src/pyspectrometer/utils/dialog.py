"""Simple dialog utilities."""


def prompt_label(title: str = "Save", prompt: str = "Label (optional):") -> str:
    """Show a one-line text-input dialog and return the entered string.

    Returns an empty string when the user cancels or leaves the field blank.
    """
    try:
        import tkinter as tk
        from tkinter import simpledialog

        root = tk.Tk()
        root.withdraw()
        root.lift()
        root.attributes("-topmost", True)
        value = simpledialog.askstring(title, prompt, parent=root) or ""
        root.destroy()
        return value.strip()
    except Exception:
        return ""


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
