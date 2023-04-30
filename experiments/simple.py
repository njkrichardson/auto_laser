import argparse 
import os 
from typing import Collection

import numpy as np 
import numpy.random as npr
import tqdm 

from cat import Cat, CatConfig
from constants import METER, SECOND
from control import RangingOracleController
from custom_logging import setup_logger
from laser import Laser, LaserConfig
from simulate import Simulator
from utils import setup_experiment_directory
from world import ClosedRoom, House

parser = argparse.ArgumentParser() 

parser.add_argument_group("visualizations")
parser.add_argument("--animate", action="store_true", help="render and save an animation of the simulation.")

def main(args): 
    # logging 
    experiment_directory: os.PathLike = setup_experiment_directory("basic")
    log = setup_logger(__name__, custom_handle=os.path.join(experiment_directory, "log.out"))

    # initialize the house
    room_width: float = 5.0 * METER 
    room_height: float = 5.0 * METER
    room: ClosedRoom = ClosedRoom(room_width, room_height)
    house: House = House(room) 

    # initialize cats 
    keesa_config: CatConfig = CatConfig(
        name="Keesa", 
        initial_position=np.array([npr.uniform(-room_width / 2., room_width / 2.), npr.uniform(-room_height / 2., room_height / 2.)]), 
        controller=RangingOracleController()
    )
    keesa: Cat = Cat(keesa_config)
    cats: Collection[Cat] = [keesa]

    # initialize lasers 
    laser_config: LaserConfig = LaserConfig(
        name="Bookerv0.0", 
        max_speed=1.0 * METER/SECOND, 
        controller=RangingOracleController(mode="avoid")
    )
    laser: Laser = Laser(laser_config)
    lasers: Collection[Laser] = [laser]

    # intialize simulator 
    simulator: Simulator = Simulator(house, cats, lasers, experiment_directory)
    log.info("configured simulator...")
    num_steps: int = 50

    for _ in tqdm.tqdm(range(num_steps)): 
        simulator.step(save_render_artifacts=True)

    log.info("finished simulation...")

    if args.animate: 
        log.info("rendering animation")
        simulator.create_animation()
        log.info("finished animation")

if __name__=="__main__": 
    args = parser.parse_args()
    main(args)