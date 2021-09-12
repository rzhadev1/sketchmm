#!/usr/bin/env pypy

import numpy as np
import time
import yaml
import sys
import os
from scipy.stats import maxwell
import os.path
import matplotlib.pyplot as plt

DEFAULT = {
    "parameters": {
        "epsilon": 1,
        "kB": 1,
        "m": 1,
        "sigma": 1,
        "timeStep": 2.0e-2
    },
    "boxSize": 10,
    "sampleInterval": 5,
    "rescaleInterval": 10,
    "targetTemp": 0.5,
    "N": 2
}


class Engine(object):

    DIM = 2
    debugging = False
    pos = np.zeros((0, 0), float)
    vel = np.zeros((0, 0), float)
    acc = np.zeros((0, 0), float)
    potential_E = 0.0
    kinetic_E = 0.0
    energy = 0.0
    instantT = 0.0
    avgT = 0.0
    totalT = 0.0
    totalP = 0.0
    avgP = 0.0
    instantP = 0.0
    sample_count = 0.0
    time = 0.0
    iterations = 0
    virial = 0.0

    def __init__(self):
        self.config = DEFAULT
        self.__init_simulation()

    def load_settings(self, settings_file):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        settings_path = os.path.join(dir_path, settings_file)
        try:
            f = open(settings_path, 'r')
            self.config = yaml.load(f, Loader=yaml.SafeLoader)
            f.close()
        except FileNotFoundError:
            print("Settings file not found.")

        self.__init_simulation()

    # optionally load data from .xyz file
    def load_positions(self, settings_file, data_file):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(dir_path, data_file)
        self.load_settings(settings_file)
        # overwrite lattice positions with loaded data
        try:
            f = open(data_path, 'r')
            length = int(f.readline())
            if self.N != length:
                raise NSizeMissmatch
            else:
                self.N = length
            print(f"{f.readline()}")
            data = np.asarray([line.split() for line in f.readlines()])
            self.pos = np.zeros((self.N, self.DIM))
            self.pos[:, 0] = data[:, 1].astype(float)
            self.pos[:, 1] = data[:, 2].astype(float)
            if self.pos.shape != (self.N, self.DIM):
                raise NSizeMissmatch
            f.close()
        except FileNotFoundError:
            print("Data file not found.")

    # randomly initialize particle velocities according to
    # maxwell-boltzmann distribution
    def __init_velocities(self):
        kB = self.config["parameters"]["kB"]
        m = self.config["parameters"]["m"]
        T = self.config["targetTemp"]
        scale = np.sqrt((kB*T/m))
        self.vel[:, 0] = np.random.normal(0.0, scale=scale, size=self.N)
        self.vel[:, 1] = np.random.normal(0.0, scale=scale, size=self.N)
        self.vel[:, 0] -= sum(self.vel[:, 0]) / float(self.N)
        self.vel[:, 1] -= sum(self.vel[:, 1]) / float(self.N)
        factor = np.sqrt(2.0 * T * self.N /
                         sum(self.vel[:, 0] ** 2 + self.vel[:, 1] ** 2))
        self.vel[:, 0] *= factor
        self.vel[:, 1] *= factor

    # calculate some properties and
    # use lattice structure for positions

    def __init_simulation(self):
        self.N = self.config["N"]
        self.volume = self.config["boxSize"] ** 2
        self.density = self.N / self.volume
        self.pos = np.zeros((self.N, self.DIM))
        self.vel = np.zeros((self.N, self.DIM))
        self.acc = np.zeros((self.N, self.DIM))

        # sample maxwell boltzmann splitribution for target temp
        # scale param a = sqrt((kB * T)/m) for maxwell boltzmann
        L = self.config["boxSize"]

        # lattice generation
        # -for a box of L x L dimensions,
        # there are sqrt(N) points that cover an L length side
        # with an L/sqrt(N) distance between lattice points

        points = int(round(np.power(self.N, (1.0 / self.DIM))))

        # if N is not a perfect square,
        # a perfect square lattice cant be generated.
        # Must increment points to include the extra particles and
        # recalculate distance between lattice points
        while(points ** 2 < self.N):
            points += 1

        split = L / float(points)

        # arrange particles
        index = 0
        for i in range(points):
            x = (0.5 + i) * split
            for j in range(points):
                if index < self.N:
                    y = (0.5 + j) * split
                    self.pos[index] = np.asarray((x, y))
                    index += 1

        self.__init_velocities()

    # # calculate energy, potential energy
    # kinetic energy, instant temperature, average temperature
    # of the system
    # (only calculate T if within the sample interval)

    def stats(self):
        self.kinetic_E = 0.0
        mass = self.config["parameters"]["m"]
        kB = self.config["parameters"]["kB"]
        for i in range(self.DIM):
            self.kinetic_E += (mass * 0.5 *
                               np.sum(self.vel[:, i] * self.vel[:, i]))
        self.energy = self.kinetic_E + self.potential_E

        self.instantT = self.kinetic_E / (self.N * kB)
        # this probably doesnt work...
        self.instantP = self.density * kB * self.avgT + self.virial
        if self.iterations % self.config["sampleInterval"] == 0:
            self.sample_count += 1
            self.totalT += self.instantT
            self.avgT = self.totalT / self.sample_count
            self.totalP += self.instantP
            self.avgP = self.totalP / self.sample_count

    def debug(self):
        if self.debugging:
            print(f"time: {self.time:.4f}")
            print(f"iterations: {self.iterations}")
            print(f"E: {self.energy}")
            print(f"avgT: {self.avgT}")
            print(f"instantT: {self.instantT}")
            print(f"avgP: {self.avgP}")
            print(f"instantP: {self.instantP}\n")

    
    # move simulation one time step further according to velocity verlet
    # apply periodic boundary conditions and temp control

    def step_forward(self):
        dt = self.config["parameters"]["timeStep"]
        box_length = self.config["boxSize"]
        self.iterations += 1
        self.time += dt
        # update positions to 2nd degree accuracy
        # and half time step for velocity
        for i in range(self.DIM):
            self.pos[:, i] += self.vel[:, i] * \
                dt + self.acc[:, i] * 0.5 * dt * dt
            # periodic boundary conditions
            # wrap coordinates around boundaries
            self.pos[:, i] = self.pos[:, i] % box_length
            self.vel[:, i] += 0.5 * self.acc[:, i] * dt

        if self.N >= 100:
            self.linked_list()
        else:
            self.force_field()
        # update velocity by another half time step
        for i in range(self.DIM):
            self.vel[:, i] += 0.5 * self.acc[:, i] * dt

        # rescale velocities if within the interval
        if self.iterations % self.config["rescaleInterval"] == 0:
            # make sure center of mass doesnt drift
            self.vel[:, 0] -= sum(self.vel[:, 0]) / float(self.N)
            self.vel[:, 1] -= sum(self.vel[:, 1]) / float(self.N)
            self.vel[:, 0] *= np.sqrt(self.config["targetTemp"]/self.avgT)
            self.vel[:, 1] *= np.sqrt(self.config["targetTemp"]/self.avgT)

        self.debug()
        self.stats()

    # apply cell linked list algorithm

    def linked_list(self):
        self.potential_E = 0.0
        self.virial = 0.0
        self.acc = np.zeros((self.N, self.DIM))
        box_length = self.config["boxSize"]
        eps = self.config["parameters"]["epsilon"]
        sig = self.config["parameters"]["sigma"]
        mass = self.config["parameters"]["m"]
        RCUT = 2.5
        cutoff = RCUT * sig
        cut_pE = 4 * eps * \
            (np.power((sig/cutoff), 12) - np.power((sig/cutoff), 6))
        # number of cells for a row
        l_cells = int(box_length // RCUT)
        # length of each cell
        cell_length = box_length / l_cells
        # point to the first atom in each cell
        cell_heads = np.full(l_cells * l_cells, fill_value=-1, dtype=np.int32)
        # points to the linked atom of the i-th atom in the same cell
        linked_list = np.zeros(self.N, dtype=np.int32)

        # linked list construction
        for i in range(self.N):
            # vector cell index
            xcell = int(self.pos[i][0] // cell_length)
            ycell = int(self.pos[i][1] // cell_length)
            # scalar cell index
            c = xcell + l_cells * ycell
            # i-th atom now points to previous head of the cell
            linked_list[i] = cell_heads[c]
            # cell head now points to this atom
            cell_heads[c] = i

        for x in range(l_cells):
            for y in range(l_cells):
                cell_index = x + l_cells * y
                # scan this cell and adjacent neighbor cells
                for nx in range(x-1, x+2):
                    for ny in range(y-1, y+2):
                        shiftx = 0.0
                        shifty = 0.0
                        # apply periodic boundary conditions
                        # if needed
                        if nx < 0:
                            shiftx = -box_length
                        elif nx >= l_cells:
                            shiftx = box_length
                        if ny < 0:
                            shifty = -box_length
                        elif ny >= l_cells:
                            shifty = box_length

                        # pull index back within range of [0, l_cells-1]
                        # if neighbor is out of bounds
                        neighbor_index = (nx + 2 * l_cells) % l_cells + \
                            (ny + 2 * l_cells) % l_cells * l_cells
                        i = cell_heads[cell_index]
                        while(i > -1):
                            j = cell_heads[neighbor_index]
                            while(j > -1):
                                if (i < j):
                                    delta_x = self.pos[i][0] - \
                                        (self.pos[j][0] + shiftx)
                                    delta_y = self.pos[i][1] - \
                                        (self.pos[j][1] + shifty)
                                    delta_x_sq = delta_x ** 2
                                    delta_y_sq = delta_y ** 2
                                    dist_sq = delta_x_sq + delta_y_sq
                                    if dist_sq < (cutoff * cutoff):
                                        inv_dist_sq = 1.0 / dist_sq
                                        para_inv = sig / dist_sq
                                        attra_term = para_inv ** 3
                                        repul_term = attra_term ** 2
                                        self.potential_E += 4.0 * eps * \
                                            (repul_term - attra_term) - cut_pE

                                        # negative gradient of LJ potential
                                        forcex = delta_x * 24.0 * eps * \
                                            ((2.0 * repul_term) -
                                             attra_term) * inv_dist_sq
                                        forcey = delta_y * 24.0 * eps * \
                                            ((2.0 * repul_term) -
                                             attra_term) * inv_dist_sq

                                        self.acc[i][0] += forcex/mass
                                        self.acc[i][1] += forcey/mass
                                        self.acc[j][0] -= forcex/mass
                                        self.acc[j][1] -= forcey/mass

                                        self.virial += np.sqrt(dist_sq) * \
                                            np.sqrt(forcex * forcex +
                                                    forcey * forcey)

                                j = linked_list[j]
                            i = linked_list[i]

        self.virial = self.virial / (self.volume * self.DIM)
        

    # calculate naive pair-wise interactions between molecules
    # or within periodic images of the system (if greater than cutoff)
    # apply forces to all particles using newton's second law

    def force_field(self):
        RCUT = 2.5
        self.potential_E = 0.0
        self.virial = 0.0
        self.acc = np.zeros((self.N, self.DIM))
        eps = self.config["parameters"]["epsilon"]
        sig = self.config["parameters"]["sigma"]
        mass = self.config["parameters"]["m"]
        # truncate at distance for LJ cutoff
        cutoff = RCUT * sig
        box_length = self.config["boxSize"]
        # shift potential energy
        cut_pE = 4 * eps * \
            (np.power((sig/cutoff), 12) - np.power((sig/cutoff), 6))

        for i in range(self.N - 1):
            for j in range(i+1, self.N):
                # apply minimum image convention
                # maximum separation between particles is L/2
                # use location of periodic image of particle j if
                # distance is > L/2.
                delta_x = self.pos[i][0] - self.pos[j][0]
                if delta_x > box_length/2:
                    delta_x -= box_length
                elif delta_x <= -box_length/2:
                    delta_x += box_length
                delta_y = self.pos[i][1] - self.pos[j][1]
                if delta_y > box_length/2:
                    delta_y -= box_length
                elif delta_y <= -box_length/2:
                    delta_y += box_length

                delta_x_sq = delta_x ** 2
                delta_y_sq = delta_y ** 2
                dist_sq = delta_x_sq + delta_y_sq
                if dist_sq < (cutoff * cutoff):
                    inv_dist_sq = 1.0 / dist_sq
                    para_inv = sig / dist_sq
                    attra_term = para_inv ** 3
                    repul_term = attra_term ** 2
                    self.potential_E += 4.0 * eps * \
                        (repul_term - attra_term) - cut_pE

                    # negative gradient of LJ potential
                    forcex = delta_x * 24.0 * eps * \
                        ((2.0 * repul_term) - attra_term) * inv_dist_sq
                    forcey = delta_y * 24.0 * eps * \
                        ((2.0 * repul_term) - attra_term) * inv_dist_sq
                    self.acc[i][0] += forcex/mass
                    self.acc[i][1] += forcey/mass
                    self.acc[j][0] -= forcex/mass
                    self.acc[j][1] -= forcey/mass

                    self.virial += np.sqrt(dist_sq) * \
                        np.sqrt(forcex * forcex + forcey * forcey)

        self.virial = self.virial / (self.volume * self.DIM)

    # reset the engine
    def reset(self):
        self.pos = np.zeros((self.N, self.DIM), float)
        self.vel = np.zeros((self.N, self.DIM), float)
        self.acc = np.zeros((self.N, self.DIM), float)
        self.potential_E = 0.0
        self.kinetic_E = 0.0
        self.energy = 0.0
        self.instantT = 0.0
        self.avgT = 0.0
        self.totalT = 0.0
        self.totalP = 0.0
        self.avgP = 0.0
        self.instantP = 0.0
        self.sample_count = 0.0
        self.time = 0.0
        self.iterations = 0
        self.virial = 0.0
        self.__init_simulation()

    # benchmark the performance of the force field functions
    def perft():
        pass


class NSizeMissmatch(Exception):
    """The number of particles in the data input does not match
    the settings file."""
    pass


if __name__ == "__main__":
    e = Engine()
    e.load_settings("./settings.yaml")
    for i in range(1000):
        e.step_forward()
