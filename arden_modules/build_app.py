"""ArdenSignals — лаунчер терминала сигналов Арденского (открывает UI в браузере)."""
import sys, os, shutil, tempfile, webbrowser

def resource(name):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)

def main():
    src = resource("arden_ui_offline.html")
    dst = os.path.join(tempfile.gettempdir(), "ArdenSignals.html")
    try:
        shutil.copyfile(src, dst); target = dst
    except Exception:
        target = src
    webbrowser.open("file:///" + target.replace("\\", "/"))

if __name__ == "__main__":
    main()
