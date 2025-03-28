# ##################################################################### #
# 16820: Computer Vision Homework 5
# Carnegie Mellon University
#
# Nov, 2023
# ##################################################################### #

import numpy as np
import matplotlib.pyplot as plt
from q1 import (
    loadData,
    estimateAlbedosNormals,
    displayAlbedosNormals,
    estimateShape,
)
from q1 import estimateShape
from utils import enforceIntegrability, plotSurface

def estimatePseudonormalsUncalibrated(I):
    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions.

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    """

    U, S, V = np.linalg.svd(I, full_matrices=False)
    B = np.diag(np.sqrt(S[:3])) @ V[:3, :]
    L = U[:, :3]@ np.diag(np.sqrt(S[:3])).T

    return B, L


def plotBasRelief(B, mu, nu, lam):
    """
    Question 2 (f)

    Make a 3D plot of of a bas-relief transformation with the given parameters.

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals

    mu : float
        bas-relief parameter

    nu : float
        bas-relief parameter

    lambda : float
        bas-relief parameter

    Returns
    -------
        None

    """

    G = np.array([[1, 0, 0], [0, 1, 0], [mu, nu, lam]])
    Bt = enforceIntegrability(B, s)
    B_bas = np.matmul(np.linalg.inv(G).T, Bt)
    albedos, normals = estimateAlbedosNormals(B_bas)
    surface = estimateShape(normals, s)
    plotSurface(surface, "Bas-Relief-mu-{}-nu-{}-lam-{}".format(mu, nu, lam))


if __name__ == "__main__":

    # Part 2 (b)
    I, L_0, s = loadData("../data/")
    print("Original Lighting Directions: ", L_0)
    B, L = estimatePseudonormalsUncalibrated(I)
    print("Estimated Lighting Directions: ", L.T)

    albedos, normals = estimateAlbedosNormals(B)
    # print("Albedos shape: ", albedos.shape)
    # print("Normals shape: ", normals.shape)
    albedos_image, normals_image = displayAlbedosNormals(albedos, normals, s)
    plt.imsave("2b-albedo.png", albedos_image, cmap="gray")
    plt.imsave("2b-normals.png", normals_image, cmap="rainbow")

    # Part 2 (d)
    surface = estimateShape(normals, s)
    # plotSurface(surface, "Uncalibrated, attempt 1")

    # Part 2 (e)
    B1 = enforceIntegrability(B, s)
    albedos, normals = estimateAlbedosNormals(B1)
    surface = estimateShape(normals, s)
    plotSurface(surface, "Uncalibrated, attempt 2")

    # Part 2 (f)
    # vary mu
    plotBasRelief(B, 0.1, 1, 1)
    plotBasRelief(B, 0.5, 1, 1)
    plotBasRelief(B, 1, 1, 1)
    # vary nu
    plotBasRelief(B, 1, 0.1, 1)
    plotBasRelief(B, 1, 0.5, 1)
    plotBasRelief(B, 1, 1, 1)
    # vary lambda
    plotBasRelief(B, 1, 1, 0.5)
    plotBasRelief(B, 1, 1, 1)
    plotBasRelief(B, 1, 1, -1)

    # flatten the surface
    plotBasRelief(B, 10, -10, 0.1)




