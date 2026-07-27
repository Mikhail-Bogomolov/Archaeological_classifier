#!/usr/bin/env python3
"""
Диагностика камеры на Orange Pi.
Запустите: python3 scripts/diagnose_camera.py
"""
import re
import subprocess
import sys
from pathlib import Path


def run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT,
                                       timeout=5).decode("utf-8", errors="ignore")
    except Exception as e:
        return f"[ошибка: {e}]"


print("=== lsusb ===")
print(run(["lsusb"]))

print("=== /dev/video* ===")
nodes = sorted(Path("/dev").glob("video*"))
print(" ".join(str(p) for p in nodes) or "(нет)")

print("\n=== v4l2-ctl --list-devices ===")
print(run(["v4l2-ctl", "--list-devices"]))

print("=== sysfs name для каждого видео-узла ===")
for node in nodes:
    name_file = Path("/sys/class/video4linux") / node.name / "name"
    try:
        name = name_file.read_text().strip()
    except OSError:
        name = "?"
    print(f"  {node}  →  {name}")

print("\n=== Попытка открыть каждый /dev/videoN через OpenCV ===")
try:
    import cv2
    for node in nodes:
        m = re.search(r"video(\d+)$", node.name)
        if not m:
            continue
        idx = int(m.group(1))
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        opened = cap.isOpened()
        frame_ok = False
        if opened:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            ok, frame = cap.read()
            frame_ok = ok and frame is not None and frame.size > 0
        cap.release()
        status = "OK (кадр есть)" if frame_ok else ("открылась, нет кадра" if opened else "не открылась")
        print(f"  /dev/video{idx}  →  {status}")
except ImportError:
    print("  cv2 не установлен")

print("\n=== Готово. Запишите вывод и передайте разработчику. ===")
