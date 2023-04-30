from abc import ABC, abstractmethod
import dataclasses
from typing import Optional

import numpy as np 

from mobile import MobileObject

@dataclasses.dataclass
class SensorState(ABC): 
    payload: np.ndarray 
    metadata: Optional[dict]

class VirtualSensor(ABC): 
    @abstractmethod 
    def reset(self) -> None: 
        raise NotImplementedError

    @abstractmethod
    def read(self) -> SensorState: 
        raise NotImplementedError

    @abstractmethod
    def write(self, state: SensorState) -> None: 
        raise NotImplementedError

class RangingOracle(VirtualSensor): 
    name: str = "RangingOracle"

    def __init__(self, subject: Optional[MobileObject], target: Optional[MobileObject]=None): 
        self._subject: MobileObject = subject 
        self._target: MobileObject = target 

    def __repr__(self) -> str: 
        return f"{self.__class__.__name__}(name={self.name}, subject={self.subject}, target={self.target})"

    def reset(self) -> None: 
        pass

    @property 
    def subject(self) -> MobileObject: 
        return self._subject 

    @property 
    def target(self) -> MobileObject: 
        try: 
            return self._target
        except AttributeError: 
            return None

    @target.setter
    def target(self, new_target: MobileObject) -> None: 
        self._target = new_target

    def read(self) -> SensorState: 
        return SensorState(self._target.position - self._subject.position, dict(name=self.name))

    def write(self, state: SensorState) -> None: 
        pass 