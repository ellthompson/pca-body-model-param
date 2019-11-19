#include "mytools.h"
#include <Eigen/SVD>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <igl/eigs.h>
//#include <igl/invert_diag.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <string>
#include <ANN/ANN.h>

#define PI 3.14159265

std::string pad_number(int number,int needed_length){

    std::string output = std::to_string(number);

    output = std::string(needed_length - output.length(), '0') + output;

    return output;
}

void decomp(Eigen::MatrixXf M, int k, Eigen::MatrixXf & eigenvectors, Eigen::MatrixXf & evalues){

    print("Starting decomp");
/*
    Spectra::DenseSymMatProd<float> op(M);
    Spectra::SymEigsSolver< float, Spectra::LARGEST_MAGN, Spectra::DenseSymMatProd<float> > eigs(&op, k, M.rows());
    eigs.init();
    int nconv = eigs.compute();
    std::cout << "nconv  " << nconv << std::endl;
    if(eigs.info() == Spectra::SUCCESSFUL){
        evalues = eigs.eigenvalues().real();
        eigenvectors = eigs.eigenvectors().real();
        std::cout << "Eigenvalues found:\n" << evalues << std::endl;
        std::cout << "Eigenvectors found:\n" << eigenvectors << std::endl;
    }
*/
    // SVD decomposition
    Eigen::BDCSVD<Eigen::MatrixXf> C_svd(M, Eigen::ComputeFullU);// | Eigen::ComputeFullV);
    Eigen::MatrixXf U = C_svd.matrixU();
    // U is going to be of size 3n x 3n
    //Eigen::MatrixXf V = C_svd.matrixV();
    evalues = C_svd.singularValues();
    eigenvectors = U;
    std::cout << " eigenvectors matrix found, size: " << eigenvectors.rows() << " by " << eigenvectors.cols() << std::endl;

}

Eigen::MatrixXf downsample(Eigen::MatrixXd V, bool reduction, int reductionSize, int & nRows){

    // downsampling the original mesh V
    // sample rows every reductionSize rows
    // cast to float to save memory

    Eigen::MatrixXf sampledMesh;
    if (reduction){
        int nSamples = nRows / reductionSize;
        nRows = nSamples; // will change size of S
        Eigen::MatrixXf sampled(nSamples,V.cols());
        sampled.setZero();
        // sample every reductionSize rows
        int j = 0;
        for (int k = 0; k < sampled.rows(); k++) {
            sampled.row(k) = V.row(j).cast<float>();
            j = j + reductionSize;
        }
        sampledMesh = sampled.transpose();
    }
    else{ // if not subsampling, just cast to float
        sampledMesh = V.transpose().cast<float>();
    }

    return sampledMesh;
}

Eigen::MatrixXd recompose_V_matrix(Eigen::MatrixXf v){

    // resize from 3n x 1 to n x 3
    Eigen::MatrixXf output = v;
    int rows = output.rows()/3;
    output.resize(3,rows);
    output.transposeInPlace();

    return output.cast<double>();

}

Eigen::MatrixXd findIdxOfNN(Eigen::MatrixXd Q, Eigen::MatrixXd findFrom, size_t numberOfNN){
    Eigen::MatrixXd IdxNNs = Eigen::MatrixXd::Zero(Q.rows(), numberOfNN);
    // build tree
    nanoflann::KDTreeEigenMatrixAdaptor< Eigen::MatrixXd > mat_index( findFrom, 100); // 50 is the max leaf
    mat_index.index->buildIndex();

    // iterate for every q in Q to find its nearest neighbor
    for (int i = 0; i < Q.rows(); i++) {
        // current q
        Eigen::RowVector3d q_i = Q.row(i);

        // set the parameters, how many neighbors to find (num_results = 2 means : finds 1 nearest neighbor, + itself)
        const size_t num_results = numberOfNN;
        std::vector<size_t>   ret_indexes(num_results);
        std::vector<double> out_dists_sqr(num_results);

        // resultSet will contain the nearest neighbors
        nanoflann::KNNResultSet<double> resultSet(num_results);

        // find nearest neighbors
        resultSet.init(ret_indexes.data(), out_dists_sqr.data());
        mat_index.index->findNeighbors(resultSet, q_i.data() , nanoflann::SearchParams(25));

        for(int j=0; j<ret_indexes.size() ; j++ ){
            IdxNNs.coeffRef(i,j) = ret_indexes[j];
        }

    }

    return IdxNNs;
}



void print(std::string mex){
    std::cout << mex << std::endl;
}
