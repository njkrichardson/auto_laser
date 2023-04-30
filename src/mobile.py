from abc import ABC, abstractmethod

import numpy as np

class MobileObject(ABC): 
    @property 
    @abstractmethod 
    def position(self) -> np.ndarray: 
        raise NotImplementedError

    @property 
    @abstractmethod 
    def velocity(self) -> np.ndarray: 
        raise NotImplementedError 

    @property 
    @abstractmethod
    def max_speed(self) -> float: 
        raise NotImplementedError 
