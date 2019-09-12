
"""
This module allows to compute fingerprints and other descriptors for compounds saved as pdb files.
This is utilized for plotting the chemical subspace of the compounds of interest

"""
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import umap, sys, os
import seaborn as sns
import argparse
import numpy as np
from sklearn.cluster import KMeans
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Fingerprints import FingerprintMols
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold
from mpl_toolkits.mplot3d import Axes3D
from mordred import Calculator
from mordred import descriptors as desc
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib
from pathos.multiprocessing import ProcessingPool as Pool
from rdkit import DataStructs
matplotlib.rcParams.update({"font.size": 22})

__author__="Ruben Canadas"
__mail__ = "rubencr14@gmail.com"
__maintainer__="Ruben Canadas"
__version__=1.0

class LigandDescriptors(object):
	

	def __init__(self, path):

		self._path = path
		self._pdbs = sorted([files for files in os.listdir(self._path) if files.endswith(".pdb")])
		self._mols = [Chem.MolFromPDBFile(os.path.join(self._path, pdb)) for pdb in self._pdbs]
		self._draw_ligands()

	def _draw_ligands(self):

		self._im_path = os.path.join(self._path, "images")
		if not os.path.exists(self._im_path):
			os.mkdir(self._im_path)
			for mol, pdb in zip(self._mols, self._pdbs):
				Draw.MolToFile(mol, os.path.join(self._im_path, "{}.png".format(pdb.split(".")[0])), size=(500, 500))

	def __str__(self):
		return "List of ligands: {}".format(self._pdbs)

	def __getitem__(self, index):
		return self._pdbs[index]

	@property
	def pdbs(self):
		return self._pdbs


	def plot_chemical_space(self, embedding, colors=None):

		if colors is None:
			colors = [0 for _ in range(len(self._pdbs))]
		elif colors is not None:
			colors = [sns.color_palette()[x] for x in colors]

		annotations=self._pdbs
		fig, ax = plt.subplots()
		images_path = self._im_path
		images = np.array([files for files in os.listdir(images_path) if files.endswith("png")])
		image_path = sorted(np.asarray(images))
		annot1 = ax.annotate("", xy=(5,5), xytext=(1000, 1000), xycoords="figure pixels",
			bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
		image = plt.imread(os.path.join(images_path, image_path[0]))
		im = OffsetImage(image, zoom=0.6)
		annot = AnnotationBbox(im, xy=(0,0), xybox=(1.2, 0.5), boxcoords="offset points")
		ax.add_artist(annot)
		annot.set_visible(False)
		annot1.set_visible(False)
		matplotlib.rcParams.update({"font.size": 22})
		with plt.style.context("seaborn-pastel"):
			sc = plt.scatter(embedding[:, 0], embedding[:, 1], c = colors, s=200, alpha=1, edgecolors="k")
			ax.set_xlabel("PCA 1", fontsize=26, labelpad=25)
			ax.set_ylabel("PCA 2", fontsize=26, labelpad=30)


		def update_annot(ind):

			pos = sc.get_offsets()[ind["ind"][0]]
			annot.xy = pos
			annot1.xy = pos
			annot1.set_text(annotations[int(ind["ind"][0])])
			ax.set_title(annotations[int(ind["ind"][0])])
			im.set_data(plt.imread(os.path.join(images_path, image_path[int(ind["ind"][0])]), format="png"))

		def hover(event):
			
			vis = annot.get_visible()
			if event.inaxes == ax:
				cont, ind = sc.contains(event)
				if cont:
					update_annot(ind)
					annot.set_visible(True)
					annot1.set_visible(False)
					fig.canvas.draw_idle()
				else:
					if vis:
						annot.set_visible(False)
						annot1.set_visible(False)
						fig.canvas.draw_idle()

		fig.canvas.mpl_connect("motion_notify_event", hover)
		plt.show()


	def project_umap(self, X, n_neighbors=9, min_dist=0.6, random_state=2019):

		reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
		return reducer.fit_transform(X)

	def project_pca(self, X, n_components=3, random_state=2019):

		pca = PCA(n_components=n_components, random_state=random_state)
		scaler = StandardScaler()
		return pca.fit_transform(scaler.fit_transform(X))


class MorganFingerprints(LigandDescriptors):
	
	def __init__(self, path):
		super(MorganFingerprints, self).__init__(path)

	def compute_descriptors(self):
		
		return self.__morgan_fingerprints()

	def __morgan_fingerprints(self):

		def get_morgan_asdict(mol):

			fingerprints = {}
			fingerprints.update(AllChem.GetMorganFingerprint(mol, 2).GetNonzeroElements())
			return fingerprints

		fps = []
		for mol in self._mols:
			fps.append(get_morgan_asdict(mol))

		vector = DictVectorizer(sparse=True, dtype=float)
		vector.fit(fps)
		fps_array = vector.transform(fps).toarray()
		#Now we get rid of low-variance features
		variance = VarianceThreshold(threshold=0.01)

		return pd.DataFrame(variance.fit_transform(fps_array))


class CompoundDescriptors(LigandDescriptors):


	def __init__(self, path):
		super(CompoundDescriptors, self).__init__(path)

	def compute_descriptor(self, mol):

		calc = Calculator(desc)
		df = calc.pandas([mol])
		return df 

	def compute_descriptors(self, num_procs=10):

		def remove_invalid_strings(df):

			"""	
			Mordred adds some invalid messages when computing descriptors. This method gets rid of them in order to use only
			numerical values
			"""

			invalid_features = set()
			for feat in df.columns.values:
				for value in df[feat].values:
					if str(value).startswith("invalid") or str(value).startswith("min") or str(value).startswith("max") or\
						 str(value).startswith("float") or str(value).startswith("divide") or str(value).startswith("True") or\
						 str(value).startswith("False"):
						invalid_features.add(feat)
			new_df = df.drop(list(invalid_features), axis=1).replace(0, np.nan).dropna(how="any", axis=1)
			return new_df

		pool = Pool(num_procs)
		dfs = pool.map(self.compute_descriptor, self._mols)
		df = pd.concat(dfs, sort=False)
		df = remove_invalid_strings(df)
		df.to_csv("mordred.csv", index=False)
		return df


def parse_args():

	parser = argparse.ArgumentParser()
	parser.add_argument("-path", "--path", type=str, help="ligands pdb file path", default=".")
	parser.add_argument("-t", "--type", help="Type of descriptors (morgan, mordred) ", type=str, default="morgan")
	parser.add_argument("-u", "--umap", help="select umap to reduce dimensionality", action="store_true")
	parser.add_argument("-p","--pca", help="select pca to reduce dimensionality", action="store_true")
	args = parser.parse_args()

	return args.path, args.type, args.umap, args.pca

def main(colors=None):

	path, descriptor, umap, pca = parse_args()
	assert descriptor in ["morgan", "mordred" ,"daylight"], "Descriptor type does not exist"
	if descriptor == "morgan":
		descriptors = MorganFingerprints(path)
	elif descriptor == "mordred":
		descriptors = CompoundDescriptors(path)

	embedding = descriptors.compute_descriptors()
	if umap:
		embedding = descriptors.project_umap(embedding)
	elif pca:
		embedding = descriptors.project_pca(embedding)
	descriptors.plot_chemical_space(embedding, colors)



if __name__=="__main__":

	#Colors for specifying mutant, wild-type and both
	colors = [0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 2, 1, 1, 0, 1, 0, 2, 0, 0, 0, 0, 0, 1, 0] 
	main(colors)
