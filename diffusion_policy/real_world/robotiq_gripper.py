"""Module to control Robotiq's grippers - tested with HAND-E"""

import math
import socket
import threading
import time
from enum import Enum
from typing import Union, Tuple, OrderedDict

# Thunder's selected deployment operating point: pendant Speed 0% / Force 0%.
# These are minimum hardware settings, not zero motion or zero gripping force.
DEFAULT_GRIPPER_SPEED = 0
DEFAULT_GRIPPER_FORCE = 0


class RobotiqGripper:
    """
    Communicates with the gripper directly, via socket with string commands, leveraging string names for variables.
    """
    # WRITE VARIABLES (CAN ALSO READ)
    ACT = 'ACT'  # act : activate (1 while activated, can be reset to clear fault status)
    GTO = 'GTO'  # gto : go to (will perform go to with the actions set in pos, for, spe)
    ATR = 'ATR'  # atr : auto-release (emergency slow move)
    ADR = 'ADR'  # adr : auto-release direction (open(1) or close(0) during auto-release)
    FOR = 'FOR'  # for : force (0-255)
    SPE = 'SPE'  # spe : speed (0-255)
    POS = 'POS'  # pos : position (0-255), 0 = open
    # READ VARIABLES
    STA = 'STA'  # status (0 = is reset, 1 = activating, 3 = active)
    PRE = 'PRE'  # position request (echo of last commanded position)
    OBJ = 'OBJ'  # object detection (0 = moving, 1 = opening contact, 2 = closing contact, 3 = at target)
    FLT = 'FLT'  # fault (0=ok, see manual for errors if not zero)

    ENCODING = 'UTF-8'  # ASCII and UTF-8 both seem to work

    class GripperStatus(Enum):
        """Gripper status reported by the gripper. The integer values have to match what the gripper sends."""
        RESET = 0
        ACTIVATING = 1
        # UNUSED = 2  # This value is currently not used by the gripper firmware
        ACTIVE = 3

    class ObjectStatus(Enum):
        """Object status reported by the gripper. The integer values have to match what the gripper sends."""
        MOVING = 0
        STOPPED_OUTER_OBJECT = 1
        STOPPED_INNER_OBJECT = 2
        AT_DEST = 3

    def __init__(self):
        """Constructor."""
        self.socket = None
        self.command_lock = threading.Lock()
        self._min_position = 0
        self._max_position = 255
        self._min_speed = 0
        self._max_speed = 255
        self._min_force = 0
        self._max_force = 255

    def connect(self, hostname: str, port: int, socket_timeout: float = 2.0) -> None:
        """Connects to a gripper at the given address.
        :param hostname: Hostname or ip.
        :param port: Port.
        :param socket_timeout: Timeout for blocking socket operations.
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(socket_timeout)
        self.socket.connect((hostname, port))

    def disconnect(self) -> None:
        """Closes the connection with the gripper."""
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def stop(self) -> None:
        """Stop finger motion without commanding an automatic release."""
        if not self._set_var(self.GTO, 0):
            raise RuntimeError('Gripper rejected stop')

    @staticmethod
    def _deadline(timeout_s):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('timeout_s must be finite and positive')
        return time.monotonic() + timeout_s

    @staticmethod
    def _remaining(deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Gripper operation timed out')
        return remaining

    def _set_vars(self, var_dict: OrderedDict[str, Union[int, float]]):
        """Sends the appropriate command via socket to set the value of n variables, and waits for its 'ack' response.
        :param var_dict: Dictionary of variables to set (variable_name, value).
        :return: True on successful reception of ack, false if no ack was received, indicating the set may not
        have been effective.
        """
        # construct unique command
        cmd = "SET"
        for variable, value in var_dict.items():
            cmd += f" {variable} {str(value)}"
        cmd += '\n'  # new line is required for the command to finish
        # atomic commands send/rcv
        with self.command_lock:
            self.socket.sendall(cmd.encode(self.ENCODING))
            data = self.socket.recv(1024)
        return self._is_ack(data)

    def _set_var(self, variable: str, value: Union[int, float]):
        """Sends the appropriate command via socket to set the value of a variable, and waits for its 'ack' response.
        :param variable: Variable to set.
        :param value: Value to set for the variable.
        :return: True on successful reception of ack, false if no ack was received, indicating the set may not
        have been effective.
        """
        return self._set_vars(OrderedDict([(variable, value)]))

    def _get_var(self, variable: str):
        """Sends the appropriate command to retrieve the value of a variable from the gripper, blocking until the
        response is received or the socket times out.
        :param variable: Name of the variable to retrieve.
        :return: Value of the variable as integer.
        """
        # atomic commands send/rcv
        with self.command_lock:
            cmd = f"GET {variable}\n"
            self.socket.sendall(cmd.encode(self.ENCODING))
            data = self.socket.recv(1024)

        # expect data of the form 'VAR x', where VAR is an echo of the variable name, and X the value
        # note some special variables (like FLT) may send 2 bytes, instead of an integer. We assume integer here
        var_name, value_str = data.decode(self.ENCODING).split()
        if var_name != variable:
            raise ValueError(f"Unexpected response {data} ({data.decode(self.ENCODING)}): does not match '{variable}'")
        value = int(value_str)
        return value

    @staticmethod
    def _is_ack(data: str):
        return data == b'ack'

    def _reset(self, timeout_s=10.0):
        """
        Reset the gripper.
        The following code is executed in the corresponding script function
        def rq_reset(gripper_socket="1"):
            rq_set_var("ACT", 0, gripper_socket)
            rq_set_var("ATR", 0, gripper_socket)

            while(not rq_get_var("ACT", 1, gripper_socket) == 0 or not rq_get_var("STA", 1, gripper_socket) == 0):
                rq_set_var("ACT", 0, gripper_socket)
                rq_set_var("ATR", 0, gripper_socket)
                sync()
            end

            sleep(0.5)
        end
        """
        deadline = self._deadline(timeout_s)
        if not self._set_var(self.ACT, 0) or not self._set_var(self.ATR, 0):
            raise RuntimeError('Gripper rejected reset')
        while (not self._get_var(self.ACT) == 0 or not self._get_var(self.STA) == 0):
            self._remaining(deadline)
            self._set_var(self.ACT, 0)
            self._set_var(self.ATR, 0)
            time.sleep(0.01)
        time.sleep(0.5)
        self._remaining(deadline)


    def activate(self, auto_calibrate: bool = True, timeout_s: float = 30.0,
                 on_calibration_sample=None):
        """Resets the activation flag in the gripper, and sets it back to one, clearing previous fault flags.
        :param auto_calibrate: Whether to calibrate the minimum and maximum positions based on actual motion.
        The following code is executed in the corresponding script function
        def rq_activate(gripper_socket="1"):
            if (not rq_is_gripper_activated(gripper_socket)):
                rq_reset(gripper_socket)

                while(not rq_get_var("ACT", 1, gripper_socket) == 0 or not rq_get_var("STA", 1, gripper_socket) == 0):
                    rq_reset(gripper_socket)
                    sync()
                end

                rq_set_var("ACT",1, gripper_socket)
            end
        end
        def rq_activate_and_wait(gripper_socket="1"):
            if (not rq_is_gripper_activated(gripper_socket)):
                rq_activate(gripper_socket)
                sleep(1.0)

                while(not rq_get_var("ACT", 1, gripper_socket) == 1 or not rq_get_var("STA", 1, gripper_socket) == 3):
                    sleep(0.1)
                end

                sleep(0.5)
            end
        end
        """
        deadline = self._deadline(timeout_s)
        if not self.is_active():
            self._reset(timeout_s=self._remaining(deadline))
            while (not self._get_var(self.ACT) == 0 or not self._get_var(self.STA) == 0):
                self._remaining(deadline)
                time.sleep(0.01)

            if not self._set_var(self.ACT, 1):
                raise RuntimeError('Gripper rejected activation')
            time.sleep(1.0)
            while (not self._get_var(self.ACT) == 1 or not self._get_var(self.STA) == 3):
                self._remaining(deadline)
                time.sleep(0.01)

        # auto-calibrate position range if desired
        if auto_calibrate:
            self.auto_calibrate(timeout_s=self._remaining(deadline), on_sample=on_calibration_sample)
        self._remaining(deadline)

    def is_active(self):
        """Returns whether the gripper is active."""
        status = self._get_var(self.STA)
        return RobotiqGripper.GripperStatus(status) == RobotiqGripper.GripperStatus.ACTIVE

    def get_min_position(self) -> int:
        """Returns the minimum position the gripper can reach (open position)."""
        return self._min_position

    def get_max_position(self) -> int:
        """Returns the maximum position the gripper can reach (closed position)."""
        return self._max_position

    def get_open_position(self) -> int:
        """Returns what is considered the open position for gripper (minimum position value)."""
        return self.get_min_position()

    def get_closed_position(self) -> int:
        """Returns what is considered the closed position for gripper (maximum position value)."""
        return self.get_max_position()

    def is_open(self):
        """Returns whether the current position is considered as being fully open."""
        return self.get_current_position() <= self.get_open_position()

    def is_closed(self):
        """Returns whether the current position is considered as being fully closed."""
        return self.get_current_position() >= self.get_closed_position()

    def get_current_position(self) -> int:
        """Returns the current position as returned by the physical hardware."""
        return self._get_var(self.POS)

    def auto_calibrate(self, log: bool = True, timeout_s: float = 20.0, on_sample=None) -> None:
        """Attempts to calibrate the open and closed positions, by slowly closing and opening the gripper.
        Requires empty fingers. An interior positioning move precedes the endpoint sweep.
        :param log: Whether to print the results to log.
        :param on_sample: Optional callback including the calibration phase.
        """
        deadline = self._deadline(timeout_s)
        def move(phase, target):
            callback = None if on_sample is None else lambda state: on_sample(dict(state, phase=phase))
            return self.move_and_wait_for_pos(target, 64, 1, timeout_s=self._remaining(deadline),
                                              on_sample=callback)

        # Physical endpoints need not equal the nominal requests 0/255. For example,
        # an already-open Thunder reports POS=3, so requesting 0 produces no motion.
        # Position inside the range first rather than relaxing stale-status checks.
        midpoint = (self.get_open_position() + self.get_closed_position()) // 2
        position, status = move('preposition', midpoint)
        if status != self.ObjectStatus.AT_DEST:
            raise RuntimeError(f'Calibration failed prepositioning empty fingers: {status.name}')
        position, status = move('open_start', self.get_open_position())
        if RobotiqGripper.ObjectStatus(status) != RobotiqGripper.ObjectStatus.AT_DEST:
            raise RuntimeError(f"Calibration failed opening to start: {str(status)}")
        open_position = position

        # try to close as far as possible, and record the number
        (position, status) = move('close', self.get_closed_position())
        if RobotiqGripper.ObjectStatus(status) != RobotiqGripper.ObjectStatus.AT_DEST:
            raise RuntimeError(f"Calibration failed because of an object: {str(status)}")
        if position > self._max_position or position <= open_position:
            raise RuntimeError('Invalid closed calibration position')
        closed_position = position

        # try to open as far as possible, and record the number
        (position, status) = move('open_finish', self.get_open_position())
        if RobotiqGripper.ObjectStatus(status) != RobotiqGripper.ObjectStatus.AT_DEST:
            raise RuntimeError(f"Calibration failed because of an object: {str(status)}")
        if position < self._min_position or abs(position - open_position) > 2:
            raise RuntimeError('Invalid open calibration position')
        # Install both bounds only after a successful, repeatable complete sweep.
        self._min_position, self._max_position = position, closed_position

        if log:
            print(f"Gripper auto-calibrated to [{self.get_min_position()}, {self.get_max_position()}]")

    def move(self, position: int, speed: int, force: int) -> Tuple[bool, int]:
        """Sends commands to start moving towards the given position, with the specified speed and force.
        :param position: Position to move to [min_position, max_position]
        :param speed: Speed to move at [min_speed, max_speed]
        :param force: Force to use [min_force, max_force]
        :return: A tuple with a bool indicating whether the action it was successfully sent, and an integer with
        the actual position that was requested, after being adjusted to the min/max calibrated range.
        """

        def clip_val(min_val, val, max_val):
            return max(min_val, min(val, max_val))

        clip_pos = clip_val(self._min_position, position, self._max_position)
        clip_spe = clip_val(self._min_speed, speed, self._max_speed)
        clip_for = clip_val(self._min_force, force, self._max_force)

        # moves to the given position with the given speed and force
        var_dict = OrderedDict([(self.POS, clip_pos), (self.SPE, clip_spe), (self.FOR, clip_for), (self.GTO, 1)])
        return self._set_vars(var_dict), clip_pos

    def move_and_wait_for_pos(self, position: int, speed: int, force: int,
                              timeout_s: float = 5.0, settle_s: float = 0.1,
                              on_sample=None) -> Tuple[int, ObjectStatus]:  # noqa
        """Sends commands to start moving towards the given position, with the specified speed and force, and
        then waits for the move to complete.
        :param position: Position to move to [min_position, max_position]
        :param speed: Speed to move at [min_speed, max_speed]
        :param force: Force to use [min_force, max_force]
        :param timeout_s: Overall deadline (individual socket calls also have a timeout).
        :param settle_s: Continuous stopped interval required for completion.
        :param on_sample: Optional callback for raw position/status/fault samples.
        :return: A tuple with an integer representing the last position returned by the gripper after it notified
        that the move had completed, a status indicating how the move ended (see ObjectStatus enum for details). Note
        that it is possible that the position was not reached, if an object was detected during motion.
        """
        deadline = self._deadline(timeout_s)
        if not math.isfinite(settle_s) or settle_s < 0:
            raise ValueError('settle_s must be finite and nonnegative')
        initial_pos = self.get_current_position()
        if self._get_var(self.GTO) == 1 and self._get_var(self.OBJ) == self.ObjectStatus.MOVING.value:
            raise RuntimeError('Gripper is already moving; stop or finish that command first')
        started_at = time.monotonic()
        set_ok, cmd_pos = self.move(position, speed, force)
        if not set_ok:
            raise RuntimeError("Failed to set variables for move.")

        seen_motion = False
        stopped_at = stopped_pos = stopped_status = None
        last_state = dict(commanded=cmd_pos, initial_position=initial_pos)
        while True:
            try:
                self._remaining(deadline)
            except TimeoutError:
                error = TimeoutError(f'Gripper move timed out: {last_state}')
                error.last_state = last_state
                raise error from None
            requested = self._get_var(self.PRE)
            final_pos = self.get_current_position()
            status = self.ObjectStatus(self._get_var(self.OBJ))
            fault = self._get_var(self.FLT)
            go_to = self._get_var(self.GTO)
            now = time.monotonic()
            seen_motion = seen_motion or (go_to == 1 and (status == self.ObjectStatus.MOVING or final_pos != initial_pos))
            last_state = dict(t=now - started_at, position=final_pos, object_status=status.value,
                              commanded=cmd_pos, requested_position=requested, fault=fault,
                              go_to=go_to, initial_position=initial_pos, seen_motion=seen_motion)
            if on_sample is not None:
                on_sample(dict(last_state))
            if fault:
                raise RuntimeError(f'Gripper fault {fault}')
            if not 0 <= final_pos <= 255 or not 0 <= requested <= 255:
                raise RuntimeError('Invalid gripper position register')
            # PRE can echo the new target while OBJ still describes the previous move.
            # A changed position also proves motion when polling missed a short move.
            no_op = (initial_pos == cmd_pos and final_pos == cmd_pos
                     and status == self.ObjectStatus.AT_DEST)
            stopped = go_to == 1 and requested == cmd_pos and (seen_motion or no_op) and status != self.ObjectStatus.MOVING
            if not stopped:
                stopped_at = stopped_pos = stopped_status = None
            elif stopped_at is None or final_pos != stopped_pos or status != stopped_status:
                stopped_at, stopped_pos, stopped_status = now, final_pos, status
            elif now - stopped_at >= settle_s:
                return final_pos, status
            time.sleep(0.005)


if __name__ == "__main__":
    # Example usage
    gripper = RobotiqGripper()
    print("Connecting to gripper...")
    gripper.connect("192.168.1.10", 63352)  # Replace with your gripper's IP and port
    print("Connected to gripper.")
    gripper.activate()

    # Move to open position
    position, status = gripper.move_and_wait_for_pos(gripper.get_open_position(), 64, 255)
    print(f"Moved to open position: {position}, Status: {status.name}")

    # Move to closed position
    position, status = gripper.move_and_wait_for_pos(gripper.get_closed_position(), 64, 255)
    print(f"Moved to closed position: {position}, Status: {status.name}")
