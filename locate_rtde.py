import rtde_control
import os
import inspect

print(f"File: {inspect.getfile(rtde_control)}")
print("\nAvailable methods in RTDEControlInterface:")
methods = [method for method in dir(rtde_control.RTDEControlInterface) if not method.startswith('_')]
for m in sorted(methods):
    print(f" - {m}")
