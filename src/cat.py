from abc import ABC, abstractmethod 
import dataclasses 
from typing import Optional

import matplotlib.pyplot as plt 
import numpy as np 

from constants import METER, SECOND
from control import ConstantController, VirtualController
from mobile import MobileObject
from sensors import RangingOracle, VirtualSensor

@dataclasses.dataclass 
class CatConfig: 
    name: Optional[str] = "anonymous"

    # physics
    max_speed: Optional[float] = 1.0 * (METER / SECOND) 
    initial_position: Optional[np.ndarray]=np.zeros(2) * METER
    initial_velocity: Optional[np.ndarray]=np.zeros(2) * (METER / SECOND)

    # control 
    controller: Optional[VirtualController]=ConstantController()

    # sensing 
    sensor: Optional[VirtualSensor]=None

class Cat(MobileObject): 
    def __init__(self, config: CatConfig): 
        self.config: CatConfig = config
        self.name: str = self.config.name 

        # physics 
        self._max_speed: float = self.config.max_speed
        self._position: np.ndarray = self.config.initial_position
        self._velocity: np.ndarray = self.config.initial_velocity
        
        # control 
        self.controller: VirtualController = self.config.controller

        # sensing 
        if self.config.sensor == None: 
            self.sensor = RangingOracle(self)

    def __repr__(self) -> str: 
        try: 
            target_name: str = self.target.name 
        except AttributeError: 
            target_name: str = None 

        return f"{self.__class__.__name__}(name={self.name}, position={self.position}, velocity={self.velocity}, max_speed={self._max_speed}, target={target_name})"

    @property 
    def target(self) -> MobileObject: 
        try: 
            return self._target 
        except AttributeError: 
            return None
    
    @target.setter 
    def target(self, new_target: MobileObject) -> None: 
        if isinstance(self.sensor, RangingOracle): 
            self.sensor.target = new_target
        self._target = new_target

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
        self._position: np.ndarray = new_position 
    
    @velocity.setter 
    def velocity(self, new_velocity: np.ndarray) -> None: 
        self._velocity: np.ndarray = new_velocity 

    def draw(self, ax) -> None: 
        ax.scatter(self.position[0], self.position[1], c="k")