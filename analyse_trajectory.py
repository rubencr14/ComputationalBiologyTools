"""

                            GROMACS ANALYSER (for molecular dynamics trajectories)

This script is for analysing the trajectories obtained by gromacs software. It will use the MDtraj module for such
analysis. There will be some flag options which the user can pick on and the files and plots will be saved in a new
created directory

                                    Done by: Ruben Canadas Rodriguez

"""

# Let's import packages

import mdtraj as md
import glob, os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools 
import matplotlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
matplotlib.rcParams.update({"font.size": 22})
plt.style.use("seaborn-pastel")

__author__="Ruben Canadas"
__mail__ = "rubencr14@gmail.com"
__maintainer__="Ruben Canadas"
__version__=1.0


class OpenFiles:

    def __init__(self, xtc_file, top_file, trr_file=None, file=None):

        self.xtc_file = xtc_file
        self.top_file = top_file
        self.trr_file = trr_file
        self.file = file       #Any file of format: pdb,xtc,trr,dcd,binpos,Netcdf,hdf5..

    def load_xtc(self):

        """

        :param xtc_file: Trajectory in xtc file
        :param top_file: Topology file of such trajectory
        :return: trajectory loaded
        """

        return md.load_xtc(self.xtc_file, self.top_file)

    def load_trr(self):

        return md.load_trr(self.trr_file, self.top_file)

    def load_trajectory(self):

        return md.load(self.file, self.top_file)

    def information(self, file):

        with open("trajectory_info.txt", "w") as outfile:
            outfile.write("Number of frames: {}\nnumber of atoms: {}\nnumber of residues: {}\number of chains: {}\n\n".format(file.n_frames,
            file.n_atoms, file.n_residues, file.n_chains))

    def number_frames(self, file):

        return file.n_frames


class directory_manager:

    def __init__(self, directory):

        self.directory = directory

    def create_directory(self):

        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
        else:
            pass

class TrajectoryProperties:

    """
    This class will have methods for computing properties of a trajectory: rmsd, distances ..
    """

    def __init__(self, traj, metric="angstrom"):

        self.traj = traj
        self.metric = metric


    def nanometer_to_angstrom(func):

        """
        This will be a decoration method
        :param func: Function to decorate
        :return:

        """

        def new_metric(self, pairs):
            return func(self,  pairs)[0] * 10

        return new_metric



    def traj_rmsd(self, reference_frame):

        return md.rmsd(self.traj, reference_frame)

   # @nanometer_to_angstrom
    def compute_distance(self, atom_pairs):

        """

        :param atom_pairs: Each row gives the indices of two atoms involved in the interaction: np.ndarray, shape=(num_pairs, 2), dtype=int
        :return: distances : np.ndarray, shape=(n_frames, num_pairs), dtype=float (The distance, in each frame, between each pair of atoms)

        """

        return md.compute_distances(self.traj, atom_pairs)


    @nanometer_to_angstrom  # Decorator for transforming array in nanometer to array in angstroms
    def compute_displacements(self, atom_pairs):

        """

        :param atom_pairs: Each row gives the indices of two atoms: np.ndarray, shape[num_pairs, 2], dtype=int
        :return:  displacements : np.ndarray, shape=[n_frames, n_pairs, 3], dtype=float32

        """

        return md.displacements(self.traj, atom_pairs)

    @nanometer_to_angstrom #Decorator for transforming array in nanometer to array in angstroms
    def compute_contacts(self, residue_pairs):

        """

        :param residue_pairs:  An array containing pairs of indices (0-indexed) of residues to compute the contacts between
        :return: distances:  np.ndarray, shape=(n_frames, n_pairs); residues_pairs: np.ndarray, shape=(n_pairs, 2)

        """

        return md.compute_contacts(self.traj, residue_pairs)

    def compute_angles(self, angle_indices):

        """

        :param angle_indices: Each row gives the indices of three atoms which together make an angle (np.ndarray, shape=(num_angles, 3), dtype=int)
        :return: The angles are in radians (np.ndarray, shape=[n_frames, n_angles], dtype=float)

        """

        return md.compute_angles(self.traj, angle_indices)

    def compute_dihedrals(self, indices):

        """

        :param indices: Each row gives the indices of four atoms which together make a dihedral angle (np.ndarray, shape=(n_dihedrals, 4), dtype=int)
        :return: dihedrals : np.ndarray, shape=(n_frames, n_dihedrals), dtype=float. The output array gives,
        in each frame from the trajectory, each of the n_dihedrals torsion angles. The angles are measured in radians

        """

        return md.compute_dihedrals(self.traj, indices)

    def compute_hbonds(self, plot=True, color=itertools.cycle(["r","b","gold", "g"]), hbonds_to_plot=[14,15,16, 17]):

        """

        """

        hbonds = md.baker_hubbard(self.traj, periodic=False)
        label = lambda hbond: "%s -- %s" % (self.traj.topology.atom(hbond[0]), self.traj.topology.atom(hbond[2]))
        dist = md.compute_distances(self.traj, hbonds[:, [0, 2]], periodic=False)
        for hbond in hbonds:
            print(label(hbond))
        if plot:
            for i in hbonds_to_plot:
                plt.hist(dist[:,i], color=next(color), label=label(hbonds[i]), alpha=0.5)
            plt.legend()
            plt.ylabel("Freq", labelpad=20)
            plt.xlabel("donor-receptor distance [nm]", labelpad=20)
            plt.show()

    
    def pca_analysis(self, scale=True, xlabel="PC1", ylabel="PC2", fontsize=20, labelpad=20):

        """

        """
        
        pca = PCA(n_components=2)
        self.traj.superpose(self.traj, 0)
        if scale:
            reduced = pca.fit_transform(StandardScaler().fit_transform(self.traj.xyz.reshape(self.traj.n_frames, self.traj.n_atoms * 3)))
        else:
            reduced = reduced = pca.fit_transform(self.traj.xyz.reshape(self.traj.n_frames, self.traj.n_atoms * 3))
        plt.scatter(reduced[:,0], reduced[:,1], marker="o", c=self.traj.time)
        plt.xlabel(xlabel, fontsize=fontsize, labelpad=labelpad)
        plt.ylabel(ylabel, fontsize=fontsize, labelpad=labelpad)
        cbar = plt.colorbar(); cbar.set_label("Time [ps]")
        plt.show()

    def compute_sasa(self, radius=0.14, mode="residue"):


        """
        Compute the solvent accessible surface area of each atom or residue in each simulation frame

        :param radius: The radius of the probe, in nm
        :param mode: In mode == atom the extracted areas are resolved peratom In mode == residue,
        this is consolidated down to the per-residue SASA by summing over the atoms in each residue
        :return: The accessible surface area of each atom or residue in every frame. If mode == atom,
        the second dimension will index the atoms in the trajectory,
        whereas if mode == residue, the second dimension will index the residues. ( np.array, shape=(n_frames, n_features)

        """

        return md.shrake_rupley(self.traj, probe_radius = radius, mode=mode)

    def compute_radius_of_gyration(self):

        """
        Compute the radius of gyration for every frame.
        :return: Rg for every frame (ndarray)

        """

        return md.compute_rg(self.traj)

    def compute_inertia_tensor(self):

        """

        Compute the inertia tensor of a trajectory.
        :return: I_ab: np.ndarray, shape=(traj.n_frames, 3, 3), dtype=float64 (Inertia tensors for each frame)

        """

        return md.compute_inertia_tensor(self.traj)


class Plotter:

    def __init__(self, x_axis, y_axis, figure_name, style="seaborn-pastel", z_axis=None, plot=False, save=True):


        self.x_axis = x_axis
        self.y_axis = y_axis
        self.z_axis = z_axis
        self.plot = plot
        self.save = save
        self.path = "."
        self.dpis = 300
        self.title = "Old triple mutant"
        self.x_label = "Steps"
        self.y_label ="Contact His195-Ser15"
        self.z_label=None
        self.cmap = "plasma"
        self.figure_name = figure_name

        plt.style.use(style)

    def scatter_plot(self):

        plt.plot(self.x_axis, self.y_axis)
        plt.title(self.title); plt.xlabel(self.x_label); plt.ylabel(self.y_label)

        if self.z_axis is not None:
            bar = plt.colorbar(); bar.set_label(self.z_label)
        if self.plot: plt.show()
        if self.save: plt.savefig(os.path.join(self.path, "md_{}_plot.png".format(self.figure_name)),dpi=self.dpis)
        plt.clf()

    def box_plot(self):


        sns.boxplot(self.y_axis, orient="v")
        plt.title(self.title);
        plt.xlabel(self.x_label, labelpad=20);
        plt.ylabel(self.y_label, labelpad=20)
        if self.plot:plt.show()
        if self.save: plt.savefig(os.path.join(self.path, "md_{}_boxplot.png".format(self.figure_name)), dpi=self.dpis)
        plt.clf()

    def superpose_plots(self, Y_array):

        """

        :param Y_array: array of Y arrays for different trajectories (for instance when comparing the effect between
        mutants)
        :return:

        """

        #TODO: we have to load another trajectory so as to be compared

        for array in Y_array:
            plt.plot(self.x_axis, array)

        if self.plot: plt.show()
        if self.save: plt.savefig(self.path, dpi=self.dpis)
        plt.clf()


def parse_args():


    parser = argparse.ArgumentParser()

    parser.add_argument("xtc", type=str, help="Trajectory in .xtc format")
    parser.add_argument("top", type=str, help="Topology file (normally in .gro format)")
    parser.add_argument("-info", "--info", help="Save informfation of trajectory in file", action="store_true")
    parser.add_argument("-distance","--distance", type=int, help="Two atoms (number) to compute it distance along the trajectory", nargs=2)
    parser.add_argument("-contact", "--contact", type=int, help="Two residues (number of residues) to compute its contacts along the trajectory", nargs=2)
    parser.add_argument("-displacement", "--displacement", type=int, help="Atom pair for computing its displacements along the tractory", nargs=2)
    parser.add_argument("-gyration", "--gyration", help="Compute the gyration at each trajectory frame", action="store_true")
    parser.add_argument("-sasa", "--sasa",help="Compute the solvent accessible surface area of each atom or residue in each simulation frame (shrake rupley method)", action="store_true")
    parser.add_argument("-plot_style","--plot_style", type=str, help="Style of the plots (default=ggplot)", default="ggplot")
    parser.add_argument("-plot", "--plot", help="Show plots", action="store_true")
    parser.add_argument("-save_plot", "--save_plot", help="Save plots in directory", action="store_true")
    parser.add_argument("-rep", "--replicas", help="Compute mean of the replicas", type=str, nargs="+")
    parser.add_argument("-p", "--pca", help="PCA analysis of the trajectory", action="store_true")
    parser.add_argument("-hb", "--hbond", help="Compute hbond distribution", action="store_true")

    args = parser.parse_args()

    return args.xtc, args.top, args.distance, args.contact, args.displacement, args.gyration , args.sasa, \
           args.plot_style, args.plot, args.save_plot, args.replicas, args.pca, args.hbond


def main():

    

    xtc, top, distance, contact, displacement, gyration, sasa, plot_style, plot, save_plot, replicas, pca, hbond = parse_args()
    xtc = OpenFiles(xtc, top)
    traj = xtc.load_xtc()
    prop = TrajectoryProperties(traj)
    number_of_frames =  xtc.number_frames(traj)
    x_axis  = np.arange(0,number_of_frames,1)

    if pca:

        prop.pca_analysis()

    if hbond:

        prop.compute_hbonds()

    if distance is not None:

        distances = prop.compute_distance([distance])
        pl = Plotter(x_axis, distances, figure_name="distance", plot=plot, save=save_plot)
        pl.scatter_plot()
        pl.box_plot()

        print("distances ", distances)

    if contact is not None:

        contacts = prop.compute_contacts([contact])
        pl = Plotter(x_axis, contacts, figure_name="contact", plot=plot, save=save_plot)
        pl.scatter_plot()
        pl.box_plot()

    if sasa:

        sasa =prop.compute_sasa()
        print("sasa ", sasa)


if __name__=="__main__":
    main()









