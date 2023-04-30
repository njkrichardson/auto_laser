from abc import ABC, abstractmethod
import dataclasses
from typing import Any, Collection, List, Optional, Sequence

import numpy as np
import numpy.random as npr

from sensors import SensorState

@dataclasses.dataclass 
class ControlSignal: 
    payload: np.ndarray 
    metadata: Optional[dict]=None

class VirtualController(ABC): 
    @property 
    @abstractmethod 
    def history(self) -> Sequence[np.ndarray]: 
        raise NotImplementedError

    @abstractmethod 
    def reset(self) -> None: 
        raise NotImplementedError

    @property 
    @abstractmethod
    def internal_state(self) -> Optional[Any]: 
        raise NotImplementedError

    @abstractmethod
    def __call__(self, state: Collection[SensorState]) -> ControlSignal: 
        raise NotImplementedError

class ConstantController(VirtualController): 
    name: str = "ContantController"

    def __init__(self, constant: Optional[np.ndarray]=np.zeros(2)): 
        self.constant = constant
        self._history: List[ControlSignal] = [ControlSignal(constant)]

    @property 
    def history(self) -> Sequence[ControlSignal]: 
        return self._history

    def reset(self) -> None: 
        pass

    @property 
    def internal_state(self) -> np.ndarray: 
        return self.constant

    def __call__(self, state: Collection[SensorState]) -> ControlSignal: 
        return ControlSignal(self.constant, dict(name=self.name))

class RandomController(VirtualController): 
    name: str = "RandomController"

    def __init__(self, gain: Optional[float]=1.0, buffer_size: Optional[int]=10):
        self.gain: float = gain
        self._buffer_size = buffer_size
        self._history: Sequence[ControlSignal] = [] 

    def reset(self) -> None: 
        pass 

    @property 
    def history(self) -> Sequence[ControlSignal]: 
        return self._history

    @property 
    def internal_state(self) -> float: 
        return self.gain

    def __call__(self, state: Collection[SensorState]) -> np.ndarray: 
        control = ControlSignal(npr.uniform(-1.0, 1.0, size=2) * self.gain, dict(name=self.name))
        if len(self._history) >= self._buffer_size: 
            self._history.pop()

        self._history.append(control)
        return control 

class RangingOracleController(VirtualController): 
    name: str = "RangingOracleController"

    def __init__(self, mode: Optional[str]="target", buffer_size: Optional[int]=1_000): 
        self._mode = mode
        self._buffer_size = buffer_size
        self._history: Sequence[ControlSignal] = [] 

    def reset(self) -> None: 
        pass 

    @property 
    def mode(self) -> str: 
        return self._mode 
    
    @mode.setter 
    def mode(self, new_mode: str) -> None: 
        assert new_mode in ["target", "avoid"]
        self._mode = new_mode

    @property 
    def internal_state(self) -> str: 
        return self.mode

    @property 
    def history(self) -> Sequence[ControlSignal]: 
        return self._history
    
    def __call__(self, state: Collection[SensorState]) -> np.ndarray: 

        if not isinstance(state, Collection): 
            state: Collection[SensorState] = [state]

        control: np.ndarray = np.zeros(2)

        for observation in state: 
            if (np.linalg.norm(observation.payload) > np.linalg.norm(control)): 
                if self.mode == "target": 
                    control = observation.payload 
                elif self.mode == "avoid": 
                    control = -observation.payload 
                else: 
                    raise NotImplementedError

        control = ControlSignal(control, dict(name=self.name, mode=self.mode))

        if len(self._history) >= self._buffer_size: 
            self._history.pop()

        self._history.append(control.payload)

        return control 