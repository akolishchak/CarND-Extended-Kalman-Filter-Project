#include "kalman_filter.h"
#include "tools.h"
#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;


void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in)
{
    x_ = x_in;
    P_ = P_in;
    F_ = F_in;
    H_ = H_in;
    R_ = R_in;
    Q_ = Q_in;
}

void KalmanFilter::Predict()
{
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z)
{
    VectorXd z_pred = H_ * x_;
    VectorXd y = z - z_pred;
    UpdateEstimate(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z)
{
    VectorXd z_pred(3);

    float px = x_(0);
    float py = x_(1);
    float vx = x_(2);
    float vy = x_(3);

    // rho
    z_pred(0) = sqrt(px*px + py*py);
    // theta
    z_pred(1) = atan2(py, px);
    // rho_dot
    z_pred(2) = ( px*vx + py*vy ) / z_pred(0);

    // measurement prediction error
    VectorXd y = z - z_pred;
    // normalize theta error
    while ( y(1) < -M_PI )
        y(1) += 2*M_PI;
    while ( y(1) > M_PI )
        y(1) -= 2*M_PI;

    UpdateEstimate(y);
}

void KalmanFilter::UpdateEstimate(const VectorXd &y)
{
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd K = P_ * Ht * S.inverse();

    //new estimate
    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}