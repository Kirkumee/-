import os, sys, subprocess, platform, site, importlib

DIR  = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(DIR, "main.py")

def run(cmd):
    subprocess.run(cmd, shell=True)

def pip_install(pkg):
    attempts = [
        [sys.executable, "-m", "pip", "install", "--user", pkg],
        [sys.executable, "-m", "pip", "install", "--user", "--break-system-packages", pkg],
        [sys.executable, "-m", "pip", "install", "--break-system-packages", pkg],
        [sys.executable, "-m", "pip", "install", pkg],
    ]
    log = []
    for cmd in attempts:
        result = subprocess.run(cmd, capture_output=True, text=True)
        log.append(f"$ {' '.join(cmd)}\n{(result.stdout or '').strip()}\n{(result.stderr or '').strip()}".strip())
        if result.returncode == 0:
            importlib.reload(site)
            return True, "\n\n".join(log)
    return False, "\n\n".join(log)

def ensure_pip():
    check = subprocess.run([sys.executable, "-m", "pip", "--version"],
                            capture_output=True, text=True)
    if check.returncode == 0:
        return True
    print("pip не найден, устанавливаем...")
    if platform.system() == "Linux":
        run("sudo apt-get update -q")
        run("sudo apt-get install -y python3-pip")
    check = subprocess.run([sys.executable, "-m", "pip", "--version"],
                            capture_output=True, text=True)
    if check.returncode != 0:
        subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"],
                        capture_output=True, text=True)
        check = subprocess.run([sys.executable, "-m", "pip", "--version"],
                                capture_output=True, text=True)
    return check.returncode == 0

if platform.system() == "Linux":
    try:
        import tkinter
    except ImportError:
        print("Устанавливаем python3-tk...")
        run("sudo apt-get install -y python3-tk")
    try:
        from PIL import ImageTk
    except ImportError:
        print("Устанавливаем python3-pil.imagetk...")
        run("sudo apt-get install -y python3-pil.imagetk")

if not ensure_pip():
    print(
        "\nНе удалось установить pip автоматически.\n"
        "Установите его вручную:\n"
        "    sudo apt-get install -y python3-pip"
    )
    sys.exit(1)

missing = []
for mod, pkg in [("numpy", "numpy"), ("scipy", "scipy"), ("sklearn", "scikit-learn"),
                  ("matplotlib", "matplotlib"), ("spectral", "spectral"), ("tifffile", "tifffile")]:
    try:
        __import__(mod)
        continue
    except ImportError:
        pass
    print(f"Устанавливаем {pkg}...")
    ok, output = pip_install(pkg)
    importlib.invalidate_caches()
    try:
        __import__(mod)
    except ImportError:
        missing.append((mod, pkg, output.strip(), ok))

if missing:
    print("\nНе удалось установить следующие зависимости:")
    for mod, pkg, output, ok in missing:
        print(f"\n--- {pkg} (модуль '{mod}') ---")
        if ok:
            print("pip сообщил об успехе, но модуль не импортируется.")
        print(output if output else "pip не выдал вывода.")
    print(
        "\nПопробуйте вручную:\n"
        f"    python3 -m pip install --break-system-packages "
        f"{' '.join(p for _, p, _, _ in missing)}"
    )
    sys.exit(1)

if platform.system() == "Linux":
    result = subprocess.getoutput("fc-list | grep -i dejavu")
    if not result.strip():
        print("Устанавливаем шрифты DejaVu...")
        run("sudo apt-get install -y fonts-dejavu fonts-dejavu-core fonts-dejavu-extra -q")
        run("fc-cache -fv 2>/dev/null")

print("Запуск приложения...")
subprocess.run([sys.executable, MAIN])
