import tkinter as tk

from app.main_window import PDFProcessorApp
from utils.log_rotation import setup_logging


def main() -> None:
    setup_logging()
    root = tk.Tk()
    PDFProcessorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
