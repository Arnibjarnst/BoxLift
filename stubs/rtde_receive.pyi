"""
RTDE Receive Interface
"""
from __future__ import annotations
import datetime
__all__: list[str] = ['RTDEReceiveInterface']
class RTDEReceiveInterface:
    def __init__(self, hostname: str, frequency: float = -1.0, variables: list[str] = [], verbose: bool = False, use_upper_range_registers: bool = False, rt_priority: int = 0) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def disconnect(self) -> None:
        ...
    def getActualCurrent(self) -> list[float]:
        """
        Returns:
            Actual joint currents
        """
    def getActualDigitalInputBits(self) -> int:
        """
        Returns:
            Current state of the digital inputs. 0-7: Standard, 8-15:
            Configurable, 16-17: Tool
        """
    def getActualDigitalOutputBits(self) -> int:
        """
        Returns:
            Current state of the digital outputs. 0-7: Standard, 8-15:
            Configurable, 16-17: Tool
        """
    def getActualExecutionTime(self) -> float:
        """
        Returns:
            Controller real-time thread execution time
        """
    def getActualJointVoltage(self) -> list[float]:
        """
        Returns:
            Actual joint voltages
        """
    def getActualMainVoltage(self) -> float:
        """
        Returns:
            Safety Control Board: Main voltage
        """
    def getActualMomentum(self) -> float:
        """
        Returns:
            Norm of Cartesian linear momentum
        """
    def getActualQ(self) -> list[float]:
        """
        Returns:
            Actual joint positions
        """
    def getActualQd(self) -> list[float]:
        """
        Returns:
            Actual joint velocities
        """
    def getActualRobotCurrent(self) -> float:
        """
        Returns:
            Safety Control Board: Robot current
        """
    def getActualRobotVoltage(self) -> float:
        """
        Returns:
            Safety Control Board: Robot voltage (48V)
        """
    def getActualTCPForce(self) -> list[float]:
        """
        Returns:
            Generalized forces in the TCP
        """
    def getActualTCPPose(self) -> list[float]:
        """
        Returns:
            Actual Cartesian coordinates of the tool: (x,y,z,rx,ry,rz), where
            rx, ry and rz is a rotation vector representation of the tool
            orientation
        """
    def getActualTCPSpeed(self) -> list[float]:
        """
        Returns:
            Actual speed of the tool given in Cartesian coordinates
        """
    def getActualToolAccelerometer(self) -> list[float]:
        """
        Returns:
            Tool x, y and z accelerometer values
        """
    def getDigitalInState(self, arg0: int) -> bool:
        ...
    def getDigitalOutState(self, arg0: int) -> bool:
        ...
    def getFtRawWrench(self) -> list[float]:
        ...
    def getJointControlOutput(self) -> list[float]:
        """
        Returns:
            Joint control currents
        """
    def getJointMode(self) -> list[int]:
        """
        Returns:
            Joint control modes
        """
    def getJointTemperatures(self) -> list[float]:
        """
        Returns:
            Temperature of each joint in degrees Celsius
        """
    def getOutputDoubleRegister(self, arg0: int) -> float:
        """
        Get the specified output double register in either lower range [18-22]
        or upper range [42-46].
        
        Parameter ``output_id``:
            the id of the register to read, current supported range is:
            [18-22] or [42-46], this can be adjusted by changing the
            RTDEReceiveInterface output recipes and by using the
            use_upper_range_registers constructor flag to switch between lower
            and upper range.
        
        Returns:
            a double from the specified output register
        """
    def getOutputIntRegister(self, arg0: int) -> int:
        """
        Get the specified output integer register in either lower range
        [18-22] or upper range [42-46].
        
        Parameter ``output_id``:
            the id of the register to read, current supported range is:
            [18-22] or [42-46], this can be adjusted by changing the
            RTDEReceiveInterface output recipes and by using the
            use_upper_range_registers constructor flag to switch between lower
            and upper range.
        
        Returns:
            an integer from the specified output register
        """
    def getPayload(self) -> float:
        ...
    def getPayloadCog(self) -> list[float]:
        ...
    def getPayloadInertia(self) -> list[float]:
        ...
    def getRobotMode(self) -> int:
        """
        Returns:
            Robot mode -1 = ROBOT_MODE_NO_CONTROLLER 0 =
            ROBOT_MODE_DISCONNECTED 1 = ROBOT_MODE_CONFIRM_SAFETY 2 =
            ROBOT_MODE_BOOTING 3 = ROBOT_MODE_POWER_OFF 4 =
            ROBOT_MODE_POWER_ON 5 = ROBOT_MODE_IDLE 6 = ROBOT_MODE_BACKDRIVE 7
            = ROBOT_MODE_RUNNING 8 = ROBOT_MODE_UPDATING_FIRMWARE
        """
    def getRobotStatus(self) -> int:
        """
        Returns:
            Robot status Bits 0-3: Is power on | Is program running | Is teach
            button pressed | Is power button pressed
        """
    def getRuntimeState(self) -> int:
        """
        Returns:
            Program state
        """
    def getSafetyMode(self) -> int:
        """
        Returns:
            Safety mode
        """
    def getSafetyStatusBits(self) -> int:
        """
        Returns:
            Safety status bits Bits 0-10: Is normal mode | Is reduced mode |
            Is protective stopped | Is recovery mode | Is safeguard stopped |
            Is system emergency stopped | Is robot emergency stopped | Is
            emergency stopped | Is violation | Is fault | Is stopped due to
            safety
        """
    def getSpeedScaling(self) -> float:
        """
        Returns:
            Speed scaling of the trajectory limiter
        """
    def getSpeedScalingCombined(self) -> float:
        """
        Get the combined speed scaling The combined speed scaling is the speed
        scaling resulting from multiplying the speed scaling with the target
        speed fraction. The combined speed scaling takes the runtime_state of
        the controller into account. If eg. a motion is paused on the teach
        pendant, and later continued, the speed scaling will be ramped up from
        zero and return to speed_scaling * target_speed_fraction when the
        runtime_state is RUNNING again.
        
        This is useful for scaling trajectories with the slider speed scaling
        currently set on the teach pendant.
        
        Returns:
            the actual combined speed scaling
        """
    def getStandardAnalogInput0(self) -> float:
        """
        Returns:
            Standard analog input 0 [A or V]
        """
    def getStandardAnalogInput1(self) -> float:
        """
        Returns:
            Standard analog input 1 [A or V]
        """
    def getStandardAnalogOutput0(self) -> float:
        """
        Returns:
            Standard analog output 0 [A or V]
        """
    def getStandardAnalogOutput1(self) -> float:
        """
        Returns:
            Standard analog output 1 [A or V]
        """
    def getTargetCurrent(self) -> list[float]:
        """
        Returns:
            Target joint currents
        """
    def getTargetMoment(self) -> list[float]:
        """
        Returns:
            Target joint moments (torques)
        """
    def getTargetQ(self) -> list[float]:
        """
        Returns:
            Target joint positions
        """
    def getTargetQd(self) -> list[float]:
        """
        Returns:
            Target joint velocities
        """
    def getTargetQdd(self) -> list[float]:
        """
        Returns:
            Target joint accelerations
        """
    def getTargetSpeedFraction(self) -> float:
        """
        Returns:
            Target speed fraction
        """
    def getTargetTCPPose(self) -> list[float]:
        """
        Returns:
            Target Cartesian coordinates of the tool: (x,y,z,rx,ry,rz), where
            rx, ry and rz is a rotation vector representation of the tool
            orientation
        """
    def getTargetTCPSpeed(self) -> list[float]:
        """
        Returns:
            Target speed of the tool given in Cartesian coordinates
        """
    def getTimestamp(self) -> float:
        """
        Returns:
            Time elapsed since the controller was started [s]
        """
    def initPeriod(self) -> datetime.timedelta:
        ...
    def isConnected(self) -> bool:
        """
        Returns:
            Connection status for RTDE, useful for checking for lost
            connection.
        """
    def isEmergencyStopped(self) -> bool:
        """
        Returns:
            a bool indicating if the robot is in 'Emergency stop'
        """
    def isProtectiveStopped(self) -> bool:
        """
        Returns:
            a bool indicating if the robot is in 'Protective stop'
        """
    def reconnect(self) -> bool:
        """
        Returns:
            Can be used to reconnect to the robot after a lost connection.
        """
    def startFileRecording(self, filename: str, variables: list[str] = []) -> bool:
        ...
    def stopFileRecording(self) -> bool:
        ...
    def waitPeriod(self, arg0: datetime.timedelta) -> None:
        ...
