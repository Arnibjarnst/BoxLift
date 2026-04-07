"""
RTDE Control Interface
"""
from __future__ import annotations
import datetime
import typing
__all__: list[str] = ['AsyncOperationStatus', 'Path', 'PathEntry', 'RTDEControlInterface']
class AsyncOperationStatus:
    def __init__(self, arg0: int) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def changeCount(self) -> int:
        ...
    def equals(self, other: AsyncOperationStatus) -> bool:
        ...
    def isAsyncOperationRunning(self) -> bool:
        ...
    def operationId(self) -> int:
        ...
    def progress(self) -> int:
        ...
    def value(self) -> int:
        ...
class Path:
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def addEntry(self, entry: PathEntry) -> None:
        ...
    def appendMovejPath(self, path: list[list[float]]) -> None:
        ...
    def appendMovelPath(self, path: list[list[float]]) -> None:
        ...
    def clear(self) -> None:
        ...
    def size(self) -> int:
        ...
    def toScriptCode(self) -> str:
        ...
    def waypoints(self) -> list[PathEntry]:
        ...
class PathEntry:
    class eMoveType:
        """
        Members:
        
          MoveJ
        
          MoveL
        
          MoveP
        
          MoveC
        """
        MoveC: typing.ClassVar[PathEntry.eMoveType]  # value = <eMoveType.MoveC: 3>
        MoveJ: typing.ClassVar[PathEntry.eMoveType]  # value = <eMoveType.MoveJ: 0>
        MoveL: typing.ClassVar[PathEntry.eMoveType]  # value = <eMoveType.MoveL: 1>
        MoveP: typing.ClassVar[PathEntry.eMoveType]  # value = <eMoveType.MoveP: 2>
        __members__: typing.ClassVar[dict[str, PathEntry.eMoveType]]  # value = {'MoveJ': <eMoveType.MoveJ: 0>, 'MoveL': <eMoveType.MoveL: 1>, 'MoveP': <eMoveType.MoveP: 2>, 'MoveC': <eMoveType.MoveC: 3>}
        def __eq__(self, other: typing.Any) -> bool:
            ...
        def __getstate__(self) -> int:
            ...
        def __hash__(self) -> int:
            ...
        def __index__(self) -> int:
            ...
        def __init__(self, value: int) -> None:
            ...
        def __int__(self) -> int:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: int) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    class ePositionType:
        """
        Members:
        
          PositionTcpPose
        
          PositionJoints
        """
        PositionJoints: typing.ClassVar[PathEntry.ePositionType]  # value = <ePositionType.PositionJoints: 1>
        PositionTcpPose: typing.ClassVar[PathEntry.ePositionType]  # value = <ePositionType.PositionTcpPose: 0>
        __members__: typing.ClassVar[dict[str, PathEntry.ePositionType]]  # value = {'PositionTcpPose': <ePositionType.PositionTcpPose: 0>, 'PositionJoints': <ePositionType.PositionJoints: 1>}
        def __eq__(self, other: typing.Any) -> bool:
            ...
        def __getstate__(self) -> int:
            ...
        def __hash__(self) -> int:
            ...
        def __index__(self) -> int:
            ...
        def __init__(self, value: int) -> None:
            ...
        def __int__(self) -> int:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: int) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    MoveC: typing.ClassVar[PathEntry.eMoveType]  # value = <eMoveType.MoveC: 3>
    MoveJ: typing.ClassVar[PathEntry.eMoveType]  # value = <eMoveType.MoveJ: 0>
    MoveL: typing.ClassVar[PathEntry.eMoveType]  # value = <eMoveType.MoveL: 1>
    MoveP: typing.ClassVar[PathEntry.eMoveType]  # value = <eMoveType.MoveP: 2>
    PositionJoints: typing.ClassVar[PathEntry.ePositionType]  # value = <ePositionType.PositionJoints: 1>
    PositionTcpPose: typing.ClassVar[PathEntry.ePositionType]  # value = <ePositionType.PositionTcpPose: 0>
    def __init__(self, move_type: ..., position_type: ..., parameters: list[float]) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def toScriptCode(self) -> str:
        ...
class RTDEControlInterface:
    class Feature:
        """
        Members:
        
          FEATURE_BASE
        
          FEATURE_TOOL
        
          FEATURE_CUSTOM
        """
        FEATURE_BASE: typing.ClassVar[RTDEControlInterface.Feature]  # value = <Feature.FEATURE_BASE: 0>
        FEATURE_CUSTOM: typing.ClassVar[RTDEControlInterface.Feature]  # value = <Feature.FEATURE_CUSTOM: 2>
        FEATURE_TOOL: typing.ClassVar[RTDEControlInterface.Feature]  # value = <Feature.FEATURE_TOOL: 1>
        __members__: typing.ClassVar[dict[str, RTDEControlInterface.Feature]]  # value = {'FEATURE_BASE': <Feature.FEATURE_BASE: 0>, 'FEATURE_TOOL': <Feature.FEATURE_TOOL: 1>, 'FEATURE_CUSTOM': <Feature.FEATURE_CUSTOM: 2>}
        def __eq__(self, other: typing.Any) -> bool:
            ...
        def __getstate__(self) -> int:
            ...
        def __hash__(self) -> int:
            ...
        def __index__(self) -> int:
            ...
        def __init__(self, value: int) -> None:
            ...
        def __int__(self) -> int:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: int) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    class Flags:
        """
        Members:
        
          FLAG_UPLOAD_SCRIPT
        
          FLAG_USE_EXT_UR_CAP
        
          FLAG_VERBOSE
        
          FLAG_UPPER_RANGE_REGISTERS
        
          FLAG_NO_WAIT
        
          FLAG_CUSTOM_SCRIPT
        
          FLAG_NO_EXT_FT
        
          FLAG_DISABLE_REMOTE_CONTROL_CHECK
        
          FLAGS_DEFAULT
        """
        FLAGS_DEFAULT: typing.ClassVar[RTDEControlInterface.Flags]  # value = <Flags.FLAG_UPLOAD_SCRIPT: 1>
        FLAG_CUSTOM_SCRIPT: typing.ClassVar[RTDEControlInterface.Flags]  # value = <Flags.FLAG_CUSTOM_SCRIPT: 32>
        FLAG_DISABLE_REMOTE_CONTROL_CHECK: typing.ClassVar[RTDEControlInterface.Flags]  # value = <Flags.FLAG_DISABLE_REMOTE_CONTROL_CHECK: 128>
        FLAG_NO_EXT_FT: typing.ClassVar[RTDEControlInterface.Flags]  # value = <Flags.FLAG_NO_EXT_FT: 64>
        FLAG_NO_WAIT: typing.ClassVar[RTDEControlInterface.Flags]  # value = <Flags.FLAG_NO_WAIT: 16>
        FLAG_UPLOAD_SCRIPT: typing.ClassVar[RTDEControlInterface.Flags]  # value = <Flags.FLAG_UPLOAD_SCRIPT: 1>
        FLAG_UPPER_RANGE_REGISTERS: typing.ClassVar[RTDEControlInterface.Flags]  # value = <Flags.FLAG_UPPER_RANGE_REGISTERS: 8>
        FLAG_USE_EXT_UR_CAP: typing.ClassVar[RTDEControlInterface.Flags]  # value = <Flags.FLAG_USE_EXT_UR_CAP: 2>
        FLAG_VERBOSE: typing.ClassVar[RTDEControlInterface.Flags]  # value = <Flags.FLAG_VERBOSE: 4>
        __members__: typing.ClassVar[dict[str, RTDEControlInterface.Flags]]  # value = {'FLAG_UPLOAD_SCRIPT': <Flags.FLAG_UPLOAD_SCRIPT: 1>, 'FLAG_USE_EXT_UR_CAP': <Flags.FLAG_USE_EXT_UR_CAP: 2>, 'FLAG_VERBOSE': <Flags.FLAG_VERBOSE: 4>, 'FLAG_UPPER_RANGE_REGISTERS': <Flags.FLAG_UPPER_RANGE_REGISTERS: 8>, 'FLAG_NO_WAIT': <Flags.FLAG_NO_WAIT: 16>, 'FLAG_CUSTOM_SCRIPT': <Flags.FLAG_CUSTOM_SCRIPT: 32>, 'FLAG_NO_EXT_FT': <Flags.FLAG_NO_EXT_FT: 64>, 'FLAG_DISABLE_REMOTE_CONTROL_CHECK': <Flags.FLAG_DISABLE_REMOTE_CONTROL_CHECK: 128>, 'FLAGS_DEFAULT': <Flags.FLAG_UPLOAD_SCRIPT: 1>}
        def __and__(self, other: typing.Any) -> typing.Any:
            ...
        def __eq__(self, other: typing.Any) -> bool:
            ...
        def __ge__(self, other: typing.Any) -> bool:
            ...
        def __getstate__(self) -> int:
            ...
        def __gt__(self, other: typing.Any) -> bool:
            ...
        def __hash__(self) -> int:
            ...
        def __index__(self) -> int:
            ...
        def __init__(self, value: int) -> None:
            ...
        def __int__(self) -> int:
            ...
        def __invert__(self) -> typing.Any:
            ...
        def __le__(self, other: typing.Any) -> bool:
            ...
        def __lt__(self, other: typing.Any) -> bool:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __or__(self, other: typing.Any) -> typing.Any:
            ...
        def __rand__(self, other: typing.Any) -> typing.Any:
            ...
        def __repr__(self) -> str:
            ...
        def __ror__(self, other: typing.Any) -> typing.Any:
            ...
        def __rxor__(self, other: typing.Any) -> typing.Any:
            ...
        def __setstate__(self, state: int) -> None:
            ...
        def __str__(self) -> str:
            ...
        def __xor__(self, other: typing.Any) -> typing.Any:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    FEATURE_BASE: typing.ClassVar[RTDEControlInterface.Feature]  # value = <Feature.FEATURE_BASE: 0>
    FEATURE_CUSTOM: typing.ClassVar[RTDEControlInterface.Feature]  # value = <Feature.FEATURE_CUSTOM: 2>
    FEATURE_TOOL: typing.ClassVar[RTDEControlInterface.Feature]  # value = <Feature.FEATURE_TOOL: 1>
    FLAGS_DEFAULT: typing.ClassVar[RTDEControlInterface.Flags]  # value = <Flags.FLAG_UPLOAD_SCRIPT: 1>
    FLAG_CUSTOM_SCRIPT: typing.ClassVar[RTDEControlInterface.Flags]  # value = <Flags.FLAG_CUSTOM_SCRIPT: 32>
    FLAG_DISABLE_REMOTE_CONTROL_CHECK: typing.ClassVar[RTDEControlInterface.Flags]  # value = <Flags.FLAG_DISABLE_REMOTE_CONTROL_CHECK: 128>
    FLAG_NO_EXT_FT: typing.ClassVar[RTDEControlInterface.Flags]  # value = <Flags.FLAG_NO_EXT_FT: 64>
    FLAG_NO_WAIT: typing.ClassVar[RTDEControlInterface.Flags]  # value = <Flags.FLAG_NO_WAIT: 16>
    FLAG_UPLOAD_SCRIPT: typing.ClassVar[RTDEControlInterface.Flags]  # value = <Flags.FLAG_UPLOAD_SCRIPT: 1>
    FLAG_UPPER_RANGE_REGISTERS: typing.ClassVar[RTDEControlInterface.Flags]  # value = <Flags.FLAG_UPPER_RANGE_REGISTERS: 8>
    FLAG_USE_EXT_UR_CAP: typing.ClassVar[RTDEControlInterface.Flags]  # value = <Flags.FLAG_USE_EXT_UR_CAP: 2>
    FLAG_VERBOSE: typing.ClassVar[RTDEControlInterface.Flags]  # value = <Flags.FLAG_VERBOSE: 4>
    def __init__(self, hostname: str, frequency: float = -1.0, flags: int = ..., ur_cap_port: int = 50002, rt_priority: int = 0) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def directTorque(self, torque: list[float], friction_comp: bool = True) -> bool:
        ...
    def disconnect(self) -> None:
        """
        Returns:
            Can be used to disconnect from the robot. To reconnect you have to
            call the reconnect() function.
        """
    def enableExternalFtSensor(self, enable: bool, sensor_mass: float = 0.0, sensor_measuring_offset: list[float] = [0.0, 0.0, 0.0], sensor_cog: list[float] = [0.0, 0.0, 0.0]) -> bool:
        ...
    def endFreedriveMode(self) -> bool:
        ...
    def endTeachMode(self) -> bool:
        """
        Set robot back in normal position control mode after freedrive mode.
        """
    def forceMode(self, arg0: list[float], arg1: list[int], arg2: list[float], arg3: int, arg4: list[float]) -> bool:
        """
        Set robot to be controlled in force mode
        
        Parameter ``task_frame``:
            A pose vector that defines the force frame relative to the base
            frame.
        
        Parameter ``selection_vector``:
            A 6d vector of 0s and 1s. 1 means that the robot will be compliant
            in the corresponding axis of the task frame
        
        Parameter ``wrench``:
            The forces/torques the robot will apply to its environment. The
            robot adjusts its position along/about compliant axis in order to
            achieve the specified force/torque. Values have no effect for non-
            compliant axes
        
        Parameter ``type``:
            An integer [1;3] specifying how the robot interprets the force
            frame. 1: The force frame is transformed in a way such that its
            y-axis is aligned with a vector pointing from the robot tcp
            towards the origin of the force frame. 2: The force frame is not
            transformed. 3: The force frame is transformed in a way such that
            its x-axis is the projection of the robot tcp velocity vector onto
            the x-y plane of the force frame.
        
        Parameter ``limits``:
            (Float) 6d vector. For compliant axes, these values are the
            maximum allowed tcp speed along/about the axis. For non-compliant
            axes, these values are the maximum allowed deviation along/about
            an axis between the actual tcp position and the one set by the
            program.
        """
    def forceModeSetDamping(self, arg0: float) -> bool:
        """
        Sets the damping parameter in force mode.
        
        Parameter ``damping``:
            Between 0 and 1, default value is 0.005
        
        A value of 1 is full damping, so the robot will decellerate quickly if
        no force is present. A value of 0 is no damping, here the robot will
        maintain the speed.
        
        The value is stored until this function is called again. Call this
        function before force mode is entered (otherwise default value will be
        used).
        """
    def forceModeSetGainScaling(self, arg0: float) -> bool:
        """
        Scales the gain in force mode.
        
        Parameter ``scaling``:
            scaling parameter between 0 and 2, default is 1.
        
        A value larger than 1 can make force mode unstable, e.g. in case of
        collisions or pushing against hard surfaces.
        
        The value is stored until this function is called again. Call this
        function before force mode is entered (otherwise default value will be
        used)
        """
    def forceModeStop(self) -> bool:
        """
        Resets the robot mode from force mode to normal operation.
        """
    def freedriveMode(self, free_axes: list[int] = [1, 1, 1, 1, 1, 1], feature: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) -> bool:
        ...
    def ftRtdeInputEnable(self, enable: bool, sensor_mass: float = 0.0, sensor_measuring_offset: list[float] = [0.0, 0.0, 0.0], sensor_cog: list[float] = [0.0, 0.0, 0.0]) -> bool:
        ...
    def getActualJointPositionsHistory(self, arg0: int) -> list[float]:
        """
        Returns the actual past angular positions of all joints.
        
        This function returns the angular positions as reported by the function "get_actual_joint_positions()" which
        indicates the number of controller time steps occurring before the current time step.
        
        An exception is thrown if indexing goes beyond the buffer size.
        
        Parameter ``steps``:
            The number of controller time steps required to go back. 0 corresponds to "get_actual_joint_positions()".
        
        Returns:
            The joint angular position vector in rad : [Base, Shoulder, Elbow, Wrist1, Wrist2, Wrist3] that was
            actual at the provided number of steps before the current time step.
        """
    def getActualToolFlangePose(self) -> list[float]:
        ...
    def getAsyncOperationProgress(self) -> int:
        """
        Reads progress information for asynchronous operations that supports
        progress feedback (such as movePath). @retval <0 Indicates that no
        async operation is running or that an async operation has finished.
        The returned values of two consecutive async operations is never
        equal. Normally the returned values are toggled between -1 and -2.
        This allows the application to clearly detect the end of an operation
        even if it is too short to see its start. That means, if the value
        returned by this function is less than 0 and is different from that
        last value returned by this function, then a new async operation has
        finished. @retval 0 Indicates that an async operation has started -
        progress 0 @retval >= 0 Indicates the progress of an async operation.
        For example, if an operation has 3 steps, the progress ranges from 0 -
        2. The progress value is updated, before a step is executed. When the
        last step has been executed, the value will change to -1 to indicate
        the end of the async operation. @deprecated The function is deprecated and only
        here for backward compatibility. Use getAsyncOperationProgressEx() instead.
        """
    def getAsyncOperationProgressEx(self) -> AsyncOperationStatus:
        """
        Returns extended async operation progress information for asynchronous
        operations that supports progress feedback (such as movePath).
        @see AsyncOperationStatus documentation for a detailed description of the
        returned status.
        """
    def getCoriolisAndCentrifugalTorques(self, q: list[float] = [], qd: list[float] = []) -> list[float]:
        ...
    def getForwardKinematics(self, q: list[float] = [], tcp_offset: list[float] = []) -> list[float]:
        """
        Calculate the forward kinematic transformation (joint space -> tool
        space) using the calibrated robot kinematics. If no joint position
        vector is provided the current joint angles of the robot arm will be
        used. If no tcp is provided the currently active tcp of the controller
        will be used.
        
        NOTICE! If you specify the tcp_offset you must also specify the q.
        
        Parameter ``q``:
            joint position vector (Optional)
        
        Parameter ``tcp_offset``:
            tcp offset pose (Optional)
        
        Returns:
            the forward kinematic transformation as a pose
        """
    def getFreedriveStatus(self) -> int:
        ...
    def getInverseKinematics(self, x: list[float], qnear: list[float] = [], max_position_error: float = 1e-10, max_orientation_error: float = 1e-10) -> list[float]:
        """
        Calculate the inverse kinematic transformation (tool space ->
        jointspace). If qnear is defined, the solution closest to qnear is
        returned.Otherwise, the solution closest to the current joint
        positions is returned. If no tcp is provided the currently active tcp
        of the controller will be used.
        
        Parameter ``x``:
            tool pose
        
        Parameter ``qnear``:
            list of joint positions (Optional)
        
        Parameter ``maxPositionError``:
            the maximum allowed positionerror (Optional)
        
        Parameter ``maxOrientationError``:
            the maximum allowed orientationerror (Optional)
        
        Returns:
            joint positions
        """
    def getInverseKinematicsHasSolution(self, x: list[float], qnear: list[float] = [], max_position_error: float = 1e-10, max_orientation_error: float = 1e-10) -> bool:
        ...
    def getJacobian(self, pos: list[float] = [], tcp: list[float] = []) -> list[float]:
        ...
    def getJacobianTimeDerivative(self, pos: list[float] = [], vel: list[float] = [], tcp: list[float] = []) -> list[float]:
        ...
    def getJointTorques(self) -> list[float]:
        """
        Returns the torques of all joints
        
        The torque on the joints, corrected by the torque needed to move the
        robot itself (gravity, friction, etc.), returned as a vector of length
        6.
        
        Returns:
            The joint torque vector in Nm: [Base, Shoulder, Elbow, Wrist1,
            Wrist2, Wrist3]
        """
    def getMassMatrix(self, q: list[float] = [], include_rotors_inertia: bool = False) -> list[float]:
        ...
    def getRobotStatus(self) -> int:
        """
        Returns:
            Robot status Bits 0-3: Is power on | Is program running | Is teach
            button pressed | Is power button pressed
            There is a synchronization gap between the three interfaces RTDE Control
            RTDE Receive and Dashboard Client. RTDE Control and RTDE Receive open
            its own RTDE connection and so the internal state is not in sync. That
            means, if RTDE Control reports, that program is running, RTDE Receive may
            still return that program is not running. The update of the Dashboard
            Client even needs more time. That means, the dashboard client still
            returns program not running after some milliseconds have passed after
            RTDE Control already reports program running.
            \\note If you work with RTDE control and receive interface and you need to
            read the robot status or program running state, then you should always
            use the getRobotStatus() function from RTDE Control if you need a status
            that is in sync with the program uploading or reuploading of this object.
        """
    def getStepTime(self) -> float:
        """
        Returns the duration of the robot time step in seconds.
        
        In every time step, the robot controller will receive measured joint
        positions and velocities from the robot, and send desired joint
        positions and velocities back to the robot. This happens with a
        predetermined frequency, in regular intervals. This interval length is
        the robot time step.
        
        Returns:
            Duration of the robot step in seconds or 0 in case of an error
        """
    def getTCPOffset(self) -> list[float]:
        """
        Gets the active tcp offset, i.e. the transformation from the output
        flange coordinate system to the TCP as a pose.
        
        Returns:
            the TCP offset as a pose
        """
    def getTargetJointAccelerations(self) -> list[float]:
        ...
    def getTargetWaypoint(self) -> list[float]:
        """
        Returns the target waypoint of the active move
        
        This is different from the target tcp pose which returns the target
        pose for each time step. The get_target_waypoint() returns the same
        target pose for movel, movej, movep or movec during the motion. It
        returns the target tcp pose, if none of the mentioned move functions
        are running.
        
        This method is useful for calculating relative movements where the
        previous move command uses blends.
        
        Returns:
            The desired waypoint TCP vector [X, Y, Z, Rx, Ry, Rz] or and empty
            vector in case of an error.
        """
    def initPeriod(self) -> datetime.timedelta:
        ...
    def isConnected(self) -> bool:
        """
        Returns:
            Connection status for RTDE, useful for checking for lost
            connection.
        """
    def isJointsWithinSafetyLimits(self, arg0: list[float]) -> bool:
        """
        Checks if the given joint position is reachable and within the current
        safety limits of the robot. This check considers joint limits (if the
        target pose is specified as joint positions), safety planes limits,
        TCP orientation deviation limits and range of the robot. If a solution
        is found when applying the inverse kinematics to the given target TCP
        pose, this pose is considered reachable
        
        Parameter ``q``:
            joint positions
        
        Returns:
            a bool indicating if the joint positions are within the safety
            limits.
        """
    def isPoseWithinSafetyLimits(self, arg0: list[float]) -> bool:
        """
        Checks if the given pose is reachable and within the current safety
        limits of the robot. It checks safety planes limits, TCP orientation
        deviation limits and range of the robot. If a solution is found when
        applying the inverse kinematics to the given target TCP pose, this
        pose is considered reachable.
        
        Parameter ``pose``:
            target pose
        
        Returns:
            a bool indicating if the pose is within the safety limits.
        """
    def isProgramRunning(self) -> bool:
        """
        Returns true if a program is running on the controller, otherwise it
        returns false
        """
    def isSteady(self) -> bool:
        """
        Checks if robot is fully at rest.
        
        True when the robot is fully at rest, and ready to accept higher
        external forces and torques, such as from industrial screwdrivers.
        
        Note: This function will always return false in modes other than the
        standard position mode, e.g. false in force and teach mode.
        
        Returns:
            True when the robot is fully at rest. Returns False otherwise.
        """
    def jogStart(self, speeds: list[float], feature: int = ..., acc: float = 0.5, custom_frame: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) -> bool:
        """
        Starts jogging with the given speed vector with respect to the given
        feature. When jogging has started, it is possible to provide new speed
        vectors by calling the jogStart() function over and over again. This
        makes it possible to use a joystick or a 3D Space Navigator to provide
        new speed vectors if the user moves the joystick or the Space
        Navigator cap.
        
        Parameter ``speed``:
            Speed vector for translation and rotation. Translation values are
            given in mm / s and rotation values in rad / s.
        
        Parameter ``feature``:
            Configures to move to move with respect to base frame
            (FEATURE_BASE), tool frame (FEATURE_TOOL) or custom frame (FEATURE_CUSTOM)
            If the feature is FEATURE_CUSTOM then the custom_frame parameter needs to
            be a valid pose.
        
        Parameter ``custom_frame``:
            The custom_frame given as pose if the selected feature
            is FEATURE_CUSTOM
        """
    def jogStop(self) -> bool:
        """
        Stops jogging that has been started start_jog
        """
    def kickWatchdog(self) -> bool:
        """
        Kicks the watchdog safeguarding the communication. Normally you would
        kick the watchdog in your control loop. Be sure to kick it as often as
        specified by the minimum frequency of the watchdog.
        """
    @typing.overload
    def moveJ(self, path: list[list[float]], asynchronous: bool = False) -> bool:
        """
        Move to each joint position specified in a path
        
        Parameter ``path``:
            with joint positions that includes acceleration, speed and blend
            for each position
        
        Parameter ``async``:
            a bool specifying if the move command should be asynchronous. If
            async is true it is possible to stop a move command using either
            the stopJ or stopL function. Default is false, this means the
            function will block until the movement has completed.
        """
    @typing.overload
    def moveJ(self, q: list[float], speed: float = 1.05, acceleration: float = 1.4, asynchronous: bool = False) -> bool:
        """
        Move to joint position (linear in joint-space)
        
        Parameter ``q``:
            joint positions
        
        Parameter ``speed``:
            joint speed of leading axis [rad/s]
        
        Parameter ``acceleration``:
            joint acceleration of leading axis [rad/s^2]
        
        Parameter ``async``:
            a bool specifying if the move command should be asynchronous. If
            async is true it is possible to stop a move command using either
            the stopJ or stopL function. Default is false, this means the
            function will block until the movement has completed.
        """
    def moveJ_IK(self, pose: list[float], speed: float = 1.05, acceleration: float = 1.4, asynchronous: bool = False) -> bool:
        """
        Move to pose (linear in joint-space)
        
        Parameter ``pose``:
            target pose
        
        Parameter ``speed``:
            joint speed of leading axis [rad/s]
        
        Parameter ``acceleration``:
            joint acceleration of leading axis [rad/s^2]
        
        Parameter ``async``:
            a bool specifying if the move command should be asynchronous. If
            async is true it is possible to stop a move command using either
            the stopJ or stopL function. Default is false, this means the
            function will block until the movement has completed.
        """
    @typing.overload
    def moveL(self, path: list[list[float]], asynchronous: bool = False) -> bool:
        """
        Move to each pose specified in a path
        
        Parameter ``path``:
            with tool poses that includes acceleration, speed and blend for
            each position
        
        Parameter ``async``:
            a bool specifying if the move command should be asynchronous. If
            async is true it is possible to stop a move command using either
            the stopJ or stopL function. Default is false, this means the
            function will block until the movement has completed.
        """
    @typing.overload
    def moveL(self, pose: list[float], speed: float = 0.25, acceleration: float = 1.2, asynchronous: bool = False) -> bool:
        """
        Move to position (linear in tool-space)
        
        Parameter ``pose``:
            target pose
        
        Parameter ``speed``:
            tool speed [m/s]
        
        Parameter ``acceleration``:
            tool acceleration [m/s^2]
        
        Parameter ``async``:
            a bool specifying if the move command should be asynchronous. If
            async is true it is possible to stop a move command using either
            the stopJ or stopL function. Default is false, this means the
            function will block until the movement has completed.
        """
    def moveL_FK(self, q: list[float], speed: float = 0.25, acceleration: float = 1.2, asynchronous: bool = False) -> bool:
        """
        Move to position (linear in tool-space)
        
        Parameter ``q``:
            joint positions
        
        Parameter ``speed``:
            tool speed [m/s]
        
        Parameter ``acceleration``:
            tool acceleration [m/s^2]
        
        Parameter ``async``:
            a bool specifying if the move command should be asynchronous. If
            async is true it is possible to stop a move command using either
            the stopJ or stopL function. Default is false, this means the
            function will block until the movement has completed.
        """
    def movePath(self, path: Path, asynchronous: bool = False) -> bool:
        ...
    def moveUntilContact(self, xd: list[float], direction: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], acceleration: float = 1.2) -> bool:
        """
        Move the robot until contact, with specified speed and contact
        detection direction.
        
        The robot will automatically retract to the initial point of contact.
        
        Parameter ``xd``:
            tool speed [m/s] (spatial vector)
        
        Parameter ``direction``:
            List of six floats. The first three elements are interpreted as a
            3D vector (in the robot base coordinate system) giving the
            direction in which contacts should be detected. If all elements of
            the list are zero, contacts from all directions are considered.
            You can also set direction=get_target_tcp_speed() in which case it
            will detect contacts in the direction of the TCP movement.
        
        Parameter ``acceleration``:
            tool position acceleration [m/s^2]
        
        Returns:
            True once the robot is in contact.
        """
    def poseTrans(self, p_from: list[float], p_from_to: list[float]) -> list[float]:
        """
        Pose transformation to move with respect to a tool or w.r.t. a custom
        feature/frame The first argument, p_from, is used to transform the
        second argument, p_from_to, and the result is then returned. This
        means that the result is the resulting pose, when starting at the
        coordinate system of p_from, and then in that coordinate system moving
        p_from_to. This function can be seen in two different views. Either
        the function transforms, that is translates and rotates, p_from_to by
        the parameters of p_from. Or the function is used to get the resulting
        pose, when first making a move of p_from and then from there, a move
        of p_from_to. If the poses were regarded as transformation matrices,
        it would look like: @verbatim T_world->to = T_world->from * T_from->to
        T_x->to = T_x->from * T_from->to @endverbatim
        
        Parameter ``p_from``:
            starting pose (spatial vector)
        
        Parameter ``p_from_to``:
            pose change relative to starting pose (spatial vector)
        
        Returns:
            resulting pose (spatial vector)
        """
    def readContactDetection(self) -> bool:
        ...
    def reconnect(self) -> bool:
        """
        Returns:
            Can be used to reconnect to the robot after a lost connection.
        """
    def reuploadScript(self) -> bool:
        """
        In the event of an error, this function can be used to resume
        operation by reuploading the RTDE control script. This will only
        happen if a script is not already running on the controller.
        """
    def sendCustomScript(self, arg0: str) -> bool:
        """
        Send a custom ur script to the controller The function enables sending
        of short scripts which was defined inline within source code. So you
        can write code like this:
        
        ```
        const std::string inline_script =
        "def script_test():\\n"
        "\\tdef test():\\n"
        "textmsg(\\"test1\\")\\n"
        "textmsg(\\"test2\\")\\n"
        "\\tend\\n"
        "\\twrite_output_integer_register(0, 1)\\n"
        "\\ttest()\\n"
        "\\ttest()\\n"
        "\\twrite_output_integer_register(0, 2)\\n"
        "end\\n"
        "run program\\n";
        bool result = rtde_c.sendCustomScript(inline_script);
        ```
        
        Returns:
            Returns true if the script has been executed successfully and
            false on timeout
        """
    def sendCustomScriptFile(self, arg0: str) -> bool:
        """
        Send a custom ur script file to the controller
        
        Parameter ``file_path``:
            the file path to the custom ur script file
        """
    def sendCustomScriptFunction(self, arg0: str, arg1: str) -> bool:
        """
        Send a custom ur script to the controller
        
        Parameter ``function_name``:
            specify a name for the custom script function
        
        Parameter ``script``:
            the custom ur script to be sent to the controller specified as a
            string, each line must be terminated with a newline. The code will
            automatically be indented with one tab to fit with the function
            body.
        """
    def servoC(self, pose: list[float], speed: float = 0.25, acceleration: float = 1.2, blend: float = 0.0) -> bool:
        """
        Servo to position (circular in tool-space). Accelerates to and moves
        with constant tool speed v.
        
        Parameter ``pose``:
            target pose
        
        Parameter ``speed``:
            tool speed [m/s]
        
        Parameter ``acceleration``:
            tool acceleration [m/s^2]
        
        Parameter ``blend``:
            blend radius (of target pose) [m]
        """
    def servoJ(self, arg0: list[float], arg1: float, arg2: float, arg3: float, arg4: float, arg5: float) -> bool:
        """
        Servo to position (linear in joint-space)
        
        Parameter ``q``:
            joint positions [rad]
        
        Parameter ``speed``:
            NOT used in current version
        
        Parameter ``acceleration``:
            NOT used in current version
        
        Parameter ``time``:
            time where the command is controlling the robot. The function is
            blocking for time t [S]
        
        Parameter ``lookahead_time``:
            time [S], range [0.03,0.2] smoothens the trajectory with this
            lookahead time
        
        Parameter ``gain``:
            proportional gain for following target position, range [100,2000]
        """
    def servoL(self, arg0: list[float], arg1: float, arg2: float, arg3: float, arg4: float, arg5: float) -> bool:
        """
        Servo to position (linear in tool-space)
        
        Parameter ``pose``:
            target pose
        
        Parameter ``speed``:
            NOT used in current version
        
        Parameter ``acceleration``:
            NOT used in current version
        
        Parameter ``time``:
            time where the command is controlling the robot. The function is
            blocking for time t [S]
        
        Parameter ``lookahead_time``:
            time [S], range [0.03,0.2] smoothens the trajectory with this
            lookahead time
        
        Parameter ``gain``:
            proportional gain for following target position, range [100,2000]
        """
    def servoStop(self, a: float = 10.0) -> bool:
        """
        Stop servo mode and decelerate the robot.
        
        Parameter ``a``:
            rate of deceleration of the tool [m/s^2]
        """
    def setCustomScriptFile(self, arg0: str) -> None:
        """
        Assign a custom script file that will be sent to device as the main
        control script. Setting an empty file_name will disable the custom
        script loading This eases debugging when modifying the control script
        because it does not require to recompile the whole library
        """
    def setExternalForceTorque(self, arg0: list[float]) -> bool:
        ...
    def setGravity(self, arg0: list[float]) -> bool:
        ...
    def setPayload(self, arg0: float, arg1: list[float]) -> bool:
        """
        Set payload
        
        Parameter ``mass``:
            Mass in kilograms
        
        Parameter ``cog``:
            Center of Gravity, a vector [CoGx, CoGy, CoGz] specifying the
            displacement (in meters) from the toolmount. If not specified the
            current CoG will be used.
        """
    def setTargetPayload(self, mass: float, cog: list[float] = [], inertia: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) -> bool:
        ...
    def setTcp(self, arg0: list[float]) -> bool:
        """
        Sets the active tcp offset, i.e. the transformation from the output
        flange coordinate system to the TCP as a pose.
        
        Parameter ``tcp_offset``:
            A pose describing the transformation of the tcp offset.
        """
    def setWatchdog(self, min_frequency: float = 10.0) -> bool:
        """
        Enable a watchdog for the communication with a specified minimum
        frequency for which an input update is expected to arrive. The
        watchdog is useful for safety critical realtime applications eg.
        servoing. The default action taken is to shutdown the control, if the
        watchdog is not kicked with the minimum frequency.
        
        Preferably you would call this function right after the
        RTDEControlInterface has been constructed.
        
        Parameter ``min_frequency``:
            The minimum frequency an input update is expected to arrive
            defaults to 10Hz.
        """
    def speedJ(self, qd: list[float], acceleration: float = 0.5, time: float = 0.0) -> bool:
        """
        Joint speed - Accelerate linearly in joint space and continue with
        constant joint speed
        
        Parameter ``qd``:
            joint speeds [rad/s]
        
        Parameter ``acceleration``:
            joint acceleration [rad/s^2] (of leading axis)
        
        Parameter ``time``:
            time [s] before the function returns (optional)
        """
    def speedL(self, xd: list[float], acceleration: float = 0.25, time: float = 0.0) -> bool:
        """
        Tool speed - Accelerate linearly in Cartesian space and continue with
        constant tool speed. The time t is optional;
        
        Parameter ``xd``:
            tool speed [m/s] (spatial vector)
        
        Parameter ``acceleration``:
            tool position acceleration [m/s^2]
        
        Parameter ``time``:
            time [s] before the function returns (optional)
        """
    def speedStop(self, a: float = 10.0) -> bool:
        """
        Stop speed mode and decelerate the robot.
        
        Parameter ``a``:
            rate of deceleration of the tool [m/s^2] if using speedL, for
            speedJ its [rad/s^2] and rate of deceleration of leading axis.
        """
    def startContactDetection(self, direction: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) -> bool:
        ...
    def stopContactDetection(self) -> bool:
        ...
    def stopJ(self, a: float = 2.0, asynchronous: bool = False) -> None:
        """
        Stop (linear in joint space) - decelerate joint speeds to zero
        
        Parameter ``a``:
            joint acceleration [rad/s^2] (rate of deceleration of the leading
            axis).
        """
    def stopL(self, a: float = 10.0, asynchronous: bool = False) -> None:
        """
        Stop (linear in tool space) - decelerate tool speed to zero
        
        Parameter ``a``:
            tool acceleration [m/s^2] (rate of deceleration of the tool)
        """
    def stopScript(self) -> None:
        """
        This function will terminate the script on controller.
        """
    def teachMode(self) -> bool:
        """
        Set robot in freedrive mode. In this mode the robot can be moved
        around by hand in the same way as by pressing the "freedrive" button.
        The robot will not be able to follow a trajectory (eg. a movej) in
        this mode.
        """
    def toolContact(self, arg0: list[float]) -> int:
        """
        Detects when a contact between the tool and an object happens.
        
        Parameter ``direction``:
            The first three elements are interpreted as a 3D vector (in the
            robot base coordinate system) giving the direction in which
            contacts should be detected. If all elements of the list are zero,
            contacts from all directions are considered.
        
        Returns:
            The returned value is the number of time steps back to just before
            the contact have started. A value larger than 0 means that a
            contact is detected. A value of 0 means no contact.
        """
    def triggerProtectiveStop(self) -> bool:
        """
        Triggers a protective stop on the robot. Can be used for testing and
        debugging.
        """
    def waitPeriod(self, arg0: datetime.timedelta) -> None:
        ...
    def zeroFtSensor(self) -> bool:
        """
        Zeroes the TCP force/torque measurement from the builtin force/torque
        sensor by subtracting the current measurement from the subsequent.
        """
