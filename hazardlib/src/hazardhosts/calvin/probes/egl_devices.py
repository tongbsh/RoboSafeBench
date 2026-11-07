"""Probe: list the EGL devices with their DRM node, PCI bus id and CUDA index.

PyBullet's eglRendererPlugin picks ``EGL_VISIBLE_DEVICES=<i>`` from the same ``eglQueryDevicesEXT`` list, so the
EGL index of a GPU is its position here. The PCI bus id comes from the DRM node (``/sys/class/drm/cardN``) and
does not depend on ``CUDA_DEVICE_ORDER``; compare it with ``nvidia-smi --query-gpu=index,pci.bus_id``.

    python -m hazardhosts.calvin.probes.egl_devices
"""
import ctypes
import os
import sys

EGL_EXTENSIONS = 0x3055
EGL_DRM_DEVICE_FILE_EXT = 0x3233
EGL_CUDA_DEVICE_NV = 0x323A


def _proc(egl, name, restype, argtypes):
    egl.eglGetProcAddress.restype = ctypes.c_void_p
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    address = egl.eglGetProcAddress(name.encode())
    if not address:
        raise RuntimeError("eglGetProcAddress(%s) failed" % name)
    return ctypes.CFUNCTYPE(restype, *argtypes)(address)


def egl_devices():
    egl = ctypes.CDLL("libEGL.so.1")
    query_devices = _proc(egl, "eglQueryDevicesEXT", ctypes.c_uint,
                          [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int)])
    query_string = _proc(egl, "eglQueryDeviceStringEXT", ctypes.c_char_p, [ctypes.c_void_p, ctypes.c_int])
    query_attrib = _proc(egl, "eglQueryDeviceAttribEXT", ctypes.c_uint,
                         [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_long)])
    devices = (ctypes.c_void_p * 32)()
    count = ctypes.c_int(0)
    if not query_devices(32, devices, ctypes.byref(count)):
        raise RuntimeError("eglQueryDevicesEXT failed")
    rows = []
    for index in range(count.value):
        device = devices[index]
        extensions = (query_string(device, EGL_EXTENSIONS) or b"").decode()
        drm = None
        if "EGL_EXT_device_drm" in extensions:
            raw = query_string(device, EGL_DRM_DEVICE_FILE_EXT)
            drm = raw.decode() if raw else None
        cuda = None
        if "EGL_NV_device_cuda" in extensions:
            value = ctypes.c_long(-1)
            if query_attrib(device, EGL_CUDA_DEVICE_NV, ctypes.byref(value)):
                cuda = int(value.value)
        pci = None
        if drm:
            sys_dev = "/sys/class/drm/%s/device" % os.path.basename(drm)
            if os.path.exists(sys_dev):
                pci = os.path.basename(os.path.realpath(sys_dev))
        rows.append({"egl_index": index, "drm": drm, "pci_bus_id": pci, "cuda_index": cuda,
                     "nvidia": "EGL_NV_device_cuda" in extensions})
    return rows


def main():
    rows = egl_devices()
    print("EGL devices: %d" % len(rows))
    for row in rows:
        print("egl_index=%(egl_index)d drm=%(drm)s pci=%(pci_bus_id)s cuda=%(cuda_index)s nvidia=%(nvidia)s" % row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
