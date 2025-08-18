import subprocess
import sys

def install_flash_attn():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flash_attn==2.8.2", "--no-build-isolation"])
        print(f"Successfully installed flash_attn")
    except subprocess.CalledProcessError as e:
        print(f"Error installing flash_attn: {e}. This demo won't work properly.")

try:
    import flash_attn
    print(f"`flash_attn` has been installed.")
except ImportError:
    print(f"`flash_attn` is NOT installed. Trying to install...")
    install_flash_attn()