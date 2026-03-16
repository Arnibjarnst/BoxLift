import os
import sys
import inspect

sys.path.append("/home/arni/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages")

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
