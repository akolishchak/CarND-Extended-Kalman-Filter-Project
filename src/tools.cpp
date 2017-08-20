#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;


VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth)
{
    VectorXd rmse(4);
    rmse << 0,0,0,0;

    // check the validity of the following inputs:
    //  * the estimation vector size should not be zero
    //  * the estimation vector size should equal ground truth vector size
    if ( estimations.size() == 0 || estimations.size() != ground_truth.size() ) {
        cout << "Invalid estimation or ground_truth data" << endl;
        return rmse;
    }

    // accumulate squared residuals
    for( unsigned int i=0; i < estimations.size(); ++i ) {

        VectorXd diff = estimations[i] - ground_truth[i];
        diff = diff.array()*diff.array();
        rmse += diff;
    }

    // calculate the mean
    rmse /= estimations.size();

    // calculate the squared root
    rmse = rmse.array().sqrt();

    // return the result
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state)
{
    MatrixXd Hj(3,4);
    // recover state parameters
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);

    // intermediate variables
    float p2 = px*px + py*py;
    float p2_sqrt = sqrt(p2);
    float p2_sqrt3 = p2*p2_sqrt;

    // check division by zero
    if ( p2 == 0 ) {
        cout << "CalculateJacobian () - Error - Division by Zero";
        return Hj;
    }

    // compute the Jacobian matrix
    Hj <<   px/p2_sqrt, py/p2_sqrt, 0, 0,
            -py/p2, px/p2, 0, 0,
            py*(vx*py - vy*px)/p2_sqrt3, px*(vy*px - vx*py)/p2_sqrt3, px/p2_sqrt, py/p2_sqrt;

    return Hj;
}
