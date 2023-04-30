import copy 
import os 
from typing import Collection, Dict, Optional, Sequence, Union

import matplotlib 
from matplotlib import animation, gridspec 
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import numpy as np 

from cat import Cat
from constants import MILLISECOND
from control import ControlSignal
from laser import Laser 
from sensors import SensorState
from world import House

class Simulator: 
    timestep_duration: float = 10.0 * MILLISECOND

    def __init__(self, house: House, cats: Collection[Cat], lasers: Collection[Laser], artifact_path: Optional[os.PathLike]=None) -> None: 
        self.current_step: int = 0 
        self.house = house
        self.artifact_path = artifact_path
        self.cats = cats
        self.lasers = lasers 

    def __repr__(self) -> str: 
        return f"{self.__class__.__name__}(house={self.house}, cats={self.cats}, lasers={self.lasers})"

    @property
    def current_time(self) -> float: 
        return self.timestep_duration * self.current_step

    def reset(self) -> None: 
        self.current_step: int = 0 
        
        for cat in self.cats: 
            cat.reset()

        for laser in self.lasers: 
            laser.reset()

    def simulate(self, num_steps: int, **kwargs) -> None: 
        for _ in range(num_steps): 
            self.step(**kwargs)

    def step(self, **kwargs) -> None:
        if kwargs.get("save_render_artifacts", False): 
            self.save_render_artifacts()

        # move the cats based on their current velocity TODO: cat position should just be clipped to walls
        for i, cat in enumerate(self.cats):
            try: 
                cat.position += cat.velocity * self.timestep_duration
                if (not self.house.inside(cat.position)): 
                    raise ValueError
            except ValueError: 
                raise ValueError(f"Collision detected: tried to move cat to position: {cat.position}")

        # move the lasers based on their current velocity TODO: laser position should just be clipped to walls
        for i, laser in enumerate(self.lasers): 
            try: 
                laser.position += laser.velocity * self.timestep_duration
                if (not self.house.inside(laser.position)): 
                    raise ValueError
            except ValueError: 
                raise ValueError(f"Collision detected: tried to move laser to position: {laser.position}")

        # (cat, laser) distance matrix
        distances: np.ndarray = np.zeros((len(self.cats), len(self.lasers)))

        for i, cat in enumerate(self.cats): 
            for j, laser in enumerate(self.lasers): 
                distances[i, j] = np.linalg.norm(cat.position - laser.position)

        # cats target the nearest laser 
        for i, cat in enumerate(self.cats): 
            nearest_laser: Laser = self.lasers[np.argmin(distances[i, :])]
            cat.target = nearest_laser

        # lasers avoid the nearest cat 
        for j, laser in enumerate(self.lasers): 
            nearest_cat: Cat = self.cats[np.argmin(distances[:, j])]
            laser.target = nearest_cat 
            
        # cats make an observation and derive a control signal from it (TODO pedantic and too much indirection... cats can handle this internally)
        for i, cat in enumerate(self.cats): 
            observation: SensorState = cat.sensor.read()
            control_signal: ControlSignal = cat.controller(observation)
            cat.velocity = control_signal.payload 

        # lasers make an observation and derive a control signal from it (TODO pedantic and too much indirection... lasers can handle this internally)
        for j, laser in enumerate(self.lasers): 
            observation: SensorState = laser.sensor.read()
            control_signal: ControlSignal = laser.controller(observation)
            laser.velocity = control_signal.payload
    
        self.current_step += 1

    def render_frame(self) -> None: 
        save_path: os.PathLike = os.path.join(self.artifact_path, f"step_{self.current_step}")

        figure, ax = plt.subplots(nrows=1, ncols=1)
        plt.title(f"Step {self.current_step}")

        self.house.draw(ax)
        
        for cat in self.cats: 
            cat.draw(ax)

        for laser in self.lasers: 
            laser.draw(ax)

        plt.xticks([])
        plt.yticks([])
        plt.savefig(save_path)
        plt.close()

    def save_render_artifacts(self) -> None: 
        if not hasattr(self, "render_artifacts"): 
            self.render_artifacts: Sequence[Dict[str, Sequence[Dict[str, np.ndarray]]]] = []

        artifact: dict = dict()
        artifact["cats"] = []
        artifact["lasers"] = []

        for cat in self.cats: 
            cat_artifact: dict = dict(cat=copy.deepcopy(cat), control=cat.controller.history)
            artifact["cats"].append(cat_artifact)

        for laser in self.lasers: 
            laser_artifact: dict = dict(laser=copy.deepcopy(laser), control=laser.controller.history)
            artifact["lasers"].append(laser_artifact)

        self.render_artifacts.append(artifact)

    def create_animation(self) -> None: 
        save_path: os.PathLike = os.path.join(self.artifact_path, "animation.mp4")
        left = 0.125  # the left side of the subplots of the figure
        right = 0.9  # the right side of the subplots of the figure
        bottom = 0.1  # the bottom of the subplots of the figure
        top = 0.9  # the top of the subplots of the figure
        wspace = 0.5  # the amount of width reserved for blank space between subplots
        hspace = 0.5  # the amount of height reserved for white space between subplots

        fig = plt.figure()
        gs_overall = gridspec.GridSpec(1, 2)
        axs = [plt.Subplot(fig, gs_i) for gs_i in gs_overall]
        subplot_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_overall[1])
        subplot_axs = [fig.add_subplot(axs[0])] + [fig.add_subplot(gs_i) for gs_i in subplot_gs]

        ax_environment = subplot_axs[0]
        ax_cats = subplot_axs[1]
        ax_lasers = subplot_axs[2]

        max_heading = np.ones(2)

        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


        def init():
            self.house.draw(ax_environment)

            ax_cats.set_xlim((0, len(self.render_artifacts) * self.timestep_duration))

            # TODO assumes just one cat! 
            last_control: np.ndarray = np.array(self.render_artifacts[-1]["cats"][-1]["control"])
            ax_cats.set_ylim(np.min(last_control) - 1., np.max(last_control) + 1.)

            ax_lasers.set_xlim(0, len(self.render_artifacts) * self.timestep_duration)
            last_control: np.ndarray = np.array(self.render_artifacts[-1]["lasers"][-1]["control"])
            ax_lasers.set_ylim(np.min(last_control) - 1., np.max(last_control) + 1.)

            fig.tight_layout()
            plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

            return fig,

        def animate(i):
            ax_environment.clear()
            ax_cats.clear()
            ax_lasers.clear()

            ax_cats.set_xlim((0, len(self.render_artifacts) * self.timestep_duration))
            last_control: np.ndarray = np.array(self.render_artifacts[i]["cats"][-1]["control"])
            ax_cats.set_ylim(np.min(last_control) - 1., np.max(last_control) + 1.)

            ax_lasers.set_xlim(0, len(self.render_artifacts) * self.timestep_duration)
            last_control: np.ndarray = np.array(self.render_artifacts[i]["lasers"][-1]["control"])
            ax_lasers.set_ylim(np.min(last_control) - 1., np.max(last_control) + 1.)

            self.house.draw(ax_environment)

            artifact: Dict[str, Collection[Union[Cat, Laser]]] = self.render_artifacts[i]

            cats: Collection[Cat] = artifact["cats"]
            lasers: Collection[Laser] = artifact["lasers"]

            for cat_artifact in cats: 
                cat = cat_artifact["cat"]
                cat.draw(ax_environment)

                control_history = np.array(cat_artifact["control"])

                ax_cats.plot(np.arange(min(i, len(control_history))) * self.timestep_duration, control_history[:i, 0])
                ax_cats.plot(np.arange(min(i, len(control_history))) * self.timestep_duration, control_history[:i, 1])

            for laser_artifact in lasers: 
                laser = laser_artifact["laser"]
                laser.draw(ax_environment)

                control_history = np.array(laser_artifact["control"])

                ax_lasers.plot(np.arange(min(i, len(control_history))) * self.timestep_duration, control_history[:i, 0])
                ax_lasers.plot(np.arange(min(i, len(control_history))) * self.timestep_duration, control_history[:i, 1])

            ax_environment.set_title(f"Environment (step: {i})")
            ax_cats.set_title("Cat Control")
            ax_lasers.set_title("Laser Control")
            fig.tight_layout()
            plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

            return fig,

        animated = animation.FuncAnimation(fig, animate, init_func=init, frames=len(self.render_artifacts), interval=1, blit=True)
        animated.save(save_path, fps=30, extra_args=['-vcodec', 'libx264'], writer='ffmpeg')