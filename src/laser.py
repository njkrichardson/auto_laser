import dataclasses 
from typing import Collection, Optional 

import matplotlib.pyplot as plt 
import numpy as np 

from constants import METER, SECOND
from control import VirtualController, ConstantController
from mobile import MobileObject
from sensors import RangingOracle, VirtualSensor

@dataclasses.dataclass
class LaserConfig: 
    name: Optional[str] = "anonymous"

    # physics
    max_speed: Optional[float] = 1.0 * (METER / SECOND) 
    initial_position: Optional[np.ndarray]=np.zeros(2) * METER
    initial_velocity: Optional[np.ndarray]=np.zeros(2) * (METER / SECOND)

    # control 
    controller: Optional[VirtualController]=ConstantController()

    # sensing 
    sensor: Optional[VirtualSensor]=None

class Laser(MobileObject): 
    def __init__(self, config: LaserConfig): 
        self.config = config
        self.reset()

    def __repr__(self) -> str: 
        try: 
            target_name: str = self.target.name 
        except AttributeError: 
            target_name: str = None 

        return f"{self.__class__.__name__}(name={self.config.name}, position={self.position}, velocity={self.velocity}, max_speed={self._max_speed}, target={target_name})"

    def reset(self) -> None: 
        self.name: str = self.config.name 
        self._position: np.ndarray = self.config.initial_position
        self._velocity: np.ndarray = self.config.initial_velocity
        self._max_speed: float = self.config.max_speed


        if self.config.sensor == None: 
            self.sensor = RangingOracle(self)
        else: 
            self.sensor.reset()

        try: 
            self.controller.reset()
        except AttributeError: 
            self.controller = self.config.controller

    @property 
    def target(self) -> MobileObject: 
        try: 
            return self._target
        except AttributeError: 
            return None

    @property 
    def position(self) -> np.ndarray: 
        return self._position 

    @property 
    def velocity(self) -> np.ndarray: 
        return self._velocity 

    @property 
    def max_speed(self) -> float: 
        return self._max_speed

    @position.setter 
    def position(self, new_position: np.ndarray) -> None: 
        self._position = new_position 

    @velocity.setter 
    def velocity(self, new_velocity: np.ndarray) -> None: 
        self._velocity = new_velocity 

    @target.setter
    def target(self, new_target: MobileObject) -> None: 
        if isinstance(self.sensor, RangingOracle): 
            self.sensor.target = new_target
        self._target = new_target

    def draw(self, ax) -> None: 
        ax.scatter(self.position[0], self.position[1], c="tab:red", marker="*", s=50)