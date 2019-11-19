#ifndef MYTOOLS
#define MYTOOLS

#define NOMINMAX

#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <igl/vertex_triangle_adjacency.h>
#include "nanoflann.hpp"
#include <igl/fit_plane.h>
#include <Eigen/SVD>

#include <random>
#include <Eigen/LU>
#include <igl/boundary_loop.h>
#include <igl/boundary_facets.h>
#include <igl/setdiff.h>
#include <igl/slice.h>
#include <igl/slice_into.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <ANN/ANN.h>

std::string pad_number(
		int number,
		int needed_length);

void addGaussNoise(
		Eigen::MatrixXd & M,
		double scale);

void decomp(
        Eigen::MatrixXf M,
        int k,
        Eigen::MatrixXf & eigenvectors,
        Eigen::MatrixXf & evalues);

Eigen::MatrixXf downsample(
        Eigen::MatrixXd V,
        bool reduction,
        int reductionSize,
        int & nRows);

Eigen::MatrixXd recompose_V_matrix(
        Eigen::MatrixXf v);

Eigen::MatrixXd findIdxOfNN(
        Eigen::MatrixXd Q,
        Eigen::MatrixXd findFrom,
        size_t numberOfNN);

ANNpointArray getANNpointsFromMatrix(
        Eigen::MatrixXd points);

void print(std::string mex);

#endif


