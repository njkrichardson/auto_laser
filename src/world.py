from abc import ABC, abstractmethod
import dataclasses 
from typing import Collection, List, Optional, Sequence

import numpy as np

from utils import normalize

@dataclasses.dataclass
class BoundarySegment: 
    endpoints: np.ndarray
    inside_normal: np.ndarray

    def ray_intersection(self, ray_origin: np.ndarray, ray_direction: np.ndarray) -> list: 
        ray_direction = normalize(ray_direction)
        
        v1: np.ndarray = ray_origin - self.endpoints[0]
        v2: np.ndarray = self.endpoints[1] - self.endpoints[0]
        v3: np.ndarray = np.array([-ray_direction[1], ray_direction[0]])


        if np.dot(v2, v3) != 0.: 
            t1: np.ndarray = np.cross(v2, v1) / np.dot(v2, v3)
            t2: np.ndarray = np.dot(v1, v3) / np.dot(v2, v3)
        else: 
            t1 = -np.inf
            t2 = -np.inf

        intersections: List[np.ndarray] = []

        if t1 >= 0.0 and t2 >= 0.0 and t2 <= 1.0:
            intersections.append(ray_origin + t1 * ray_direction)

        return intersections

    def translate(self, translation: np.ndarray) -> None: 
        self.endpoints += translation

    def draw(self, ax) -> None: 
        ax.plot(self.endpoints[0], self.endpoints[1], c="k")

class VirtualRoom(ABC): 
    @abstractmethod
    def inside(self, point: np.ndarray) -> bool: 
        raise NotImplementedError

    @abstractmethod 
    def distance_to_boundary(self, point: np.ndarray, direction: np.ndarray) -> float: 
        raise NotImplementedError

    @abstractmethod 
    def draw(self, ax) -> None: 
        raise NotImplementedError

class ClosedRoom(VirtualRoom):
    def __init__(self, width: float, height: float): 
        self._width: float = width 
        self._height: float = height

        self.half_width: float = self._width / 2. 
        self.half_height: float = self._height / 2.

        self.bottom_left_corner: np.ndarray = np.array([-self.half_width, -self.half_height])
        self.bottom_right_corner: np.ndarray = np.array([self.half_width, -self.half_height])
        self.top_right_corner: np.ndarray = np.array([self.half_width, self.half_height])
        self.top_left_corner: np.ndarray = np.array([-self.half_width, self.half_height])
        self.origin = np.zeros(2)


        up: np.ndarray = np.array([0., 1.])
        down: np.ndarray = np.array([0., -1.])
        right: np.ndarray = np.array([1., 0.])
        left: np.ndarray = np.array([-1., 0.])

        self._walls: Sequence[BoundarySegment] = [
            BoundarySegment(endpoints=np.array([self.bottom_left_corner, self.bottom_right_corner]), inside_normal=up), 
            BoundarySegment(endpoints=np.array([self.bottom_right_corner, self.top_right_corner]), inside_normal=left), 
            BoundarySegment(endpoints=np.array([self.top_right_corner, self.top_left_corner]), inside_normal=down), 
            BoundarySegment(endpoints=np.array([self.top_left_corner, self.bottom_left_corner]), inside_normal=right), 
        ]

    @property 
    def width(self) -> float: 
        return self._width 

    @property 
    def height(self) -> float: 
        return self._height

    def translate(self, translation: np.ndarray) -> None: 
        self.origin = translation 
        for wall in self._walls: 
            wall.translate(translation)

    def __repr__(self) -> str: 
        return f"{self.__class__.__name__}(origin={self.origin}, width={self.width}, heigh={self.height})"

    def inside(self, point: np.ndarray) -> bool: 
        within_width: bool = (((point[0] - self.origin[0]) < self.half_width) and ((point[0] - self.origin[0]) > -self.half_width))
        within_height: bool = (((point[1] - self.origin[1]) < self.half_height) and ((point[1] - self.origin[1]) > -self.half_height))
        return (within_width and within_height)

    def distance_to_boundary(self, point: np.ndarray, direction: np.ndarray) -> float: 
        distances: list = [] 

        for wall in self._walls: 
            intersection = wall.ray_intersection(point, direction)

            if len(intersection) != 0: 
                distances.append(np.linalg.norm(intersection[0] - point))

        if len(distances) > 0: 
            return min(distances)
        else: 
            return [np.inf]

    def draw(self, ax) -> None: 
        for wall in self._walls: 
            wall.draw(ax)

class House: 
    def __init__(self, rooms: Collection[VirtualRoom]): 
        if not isinstance(rooms, Collection): 
            self._rooms = [rooms]
        else: 
            self._rooms = rooms

    def __repr__(self) -> str: 
        return ' '.join([self._rooms.__repr__()])

    def draw(self, ax): 
        for room in self._rooms: 
            room.draw(ax)

    def inside(self, point: np.ndarray) -> bool: 
        is_inside: bool = False

        for room in self._rooms: 
            if room.inside(point): is_inside = True

        return is_inside

    def distance_to_boundary(self, point: np.ndarray, direction: np.ndarray) -> float: 
        distance: float = np.inf

        for room in self._rooms: 
            intra_room_distance: float = room.distance_to_boundary(point, direction) 
            if (intra_room_distance < distance): distance = intra_room_distance

        return distance 
