import os
import sys
import inspect

print("Python path:")
for p in sys.path:
    print(f" - {p}")

try:
    import rtde_control
    print(f"\nFile: {inspect.getfile(rtde_control)}")
    print("\nAvailable methods in RTDEControlInterface:")
    methods = [method for method in dir(rtde_control.RTDEControlInterface) if not method.startswith('_')]
    for m in sorted(methods):
        print(f" - {m}")
except ImportError as e:
    print(f"\nImport failed: {e}")
