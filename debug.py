import sys
import subprocess

def installed_packages():
    print("Python executable:", sys.executable)
    print("Installed packages:")
    subprocess.run([sys.executable, "-m", "pip", "list"])

installed_packages()