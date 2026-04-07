"""
RTDE IO Interface
"""
from __future__ import annotations
__all__: list[str] = ['RTDEIOInterface']
class RTDEIOInterface:
    def __init__(self, hostname: str, verbose: bool = False, use_upper_range_registers: bool = False) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def disconnect(self) -> None:
        ...
    def reconnect(self) -> bool:
        """
        Returns:
            Can be used to reconnect to the robot after a lost connection.
        """
    def setAnalogOutputCurrent(self, arg0: int, arg1: float) -> bool:
        """
        Set Analog output current
        
        Parameter ``output_id``:
            The number (id) of the output, integer: [0:1]
        
        Parameter ``current_ratio``:
            current set as a (ratio) of the current span [0..1], 1 means full
            current.
        """
    def setAnalogOutputVoltage(self, arg0: int, arg1: float) -> bool:
        """
        Set Analog output voltage
        
        Parameter ``output_id``:
            The number (id) of the output, integer: [0:1]
        
        Parameter ``voltage_ratio``:
            voltage set as a (ratio) of the voltage span [0..1], 1 means full
            voltage.
        """
    def setConfigurableDigitalOut(self, arg0: int, arg1: bool) -> bool:
        """
        Set configurable digital output signal level
        
        Parameter ``output_id``:
            The number (id) of the output, integer: [0:7]
        
        Parameter ``signal_level``:
            The signal level. (boolean)
        """
    def setInputDoubleRegister(self, arg0: int, arg1: float) -> bool:
        """
        Set the specified input double register in either lower range [18-22]
        or upper range [42-46].
        
        Parameter ``input_id``:
            the id of the register to set, current supported range is: [18-22]
            or [42-46], this can be adjusted by changing the
            RTDEControlInterface input recipes and by using the
            use_upper_range_registers constructor flag to switch between lower
            and upper range.
        
        Parameter ``value``:
            the desired double value
        
        Returns:
            true if the register is successfully set, false otherwise.
        """
    def setInputIntRegister(self, arg0: int, arg1: int) -> bool:
        """
        Set the specified input integer register in either lower range [18-22]
        or upper range [42-46].
        
        Parameter ``input_id``:
            the id of the register to set, current supported range is: [18-22]
            or [42-46], this can be adjusted by changing the
            RTDEControlInterface input recipes and by using the
            use_upper_range_registers constructor flag to switch between lower
            and upper range.
        
        Parameter ``value``:
            the desired integer value
        
        Returns:
            true if the register is successfully set, false otherwise.
        """
    def setSpeedSlider(self, arg0: float) -> bool:
        """
        Set the speed slider on the controller
        
        Parameter ``speed``:
            set the speed slider on the controller as a fraction value between
            0 and 1 (1 is 100%)
        """
    def setStandardDigitalOut(self, arg0: int, arg1: bool) -> bool:
        """
        Set standard digital output signal level
        
        Parameter ``output_id``:
            The number (id) of the output, integer: [0:7]
        
        Parameter ``signal_level``:
            The signal level. (boolean)
        """
    def setToolDigitalOut(self, arg0: int, arg1: bool) -> bool:
        """
        Set tool digital output signal level
        
        Parameter ``output_id``:
            The number (id) of the output, integer: [0:1]
        
        Parameter ``signal_level``:
            The signal level. (boolean)
        """
