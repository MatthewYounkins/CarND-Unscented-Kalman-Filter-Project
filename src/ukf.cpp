#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
* Initializes Unscented Kalman filter
*/
UKF::UKF() {
	// if this is false, laser measurements will be ignored (except during init)
	use_laser_ = true;

	// if this is false, radar measurements will be ignored (except during init)
	use_radar_ = true;

	// state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
	// State dimension  
	n_x_ = 5;
	n_aug_ = n_x_ + 2;
	lambda_ = 3 - n_aug_;  //Sigma spreading parameter
	n_sigma_points_ = 2 * n_x_ + 1;
	x_ = VectorXd(n_x_);
	time_us_ = 0.0;
	// initial covariance matrix
	P_ = MatrixXd(n_x_, n_x_); 

	// Process noise standard deviation longitudinal acceleration in m/s^2
	std_a_ = 2.1;  //originally 30

	// Process noise standard deviation yaw acceleration in rad/s^2
	std_yawdd_ = 0.9;  //originally 30

	// Laser measurement noise standard deviation position1 in m
	std_laspx_ = 0.15;

	// Laser measurement noise standard deviation position2 in m
	std_laspy_ = 0.15;

	// Radar measurement noise standard deviation radius in m
	std_radr_ = 0.3;

	// Radar measurement noise standard deviation angle in rad
	std_radphi_ = 0.03;

	// Radar measurement noise standard deviation radius change in m/s
	std_radrd_ = 0.3;

	//set state dimension
	is_initialized_ = false;

	weights_ = VectorXd(n_sigma_points_); //fill weights with something? TODO

	rRadar_ = MatrixXd(3, 3);
	rLaser_ = MatrixXd(2, 2);
	//move to measurement?
	rRadar_ << std_radr_ * std_radr_, 0, 0,
		0, std_radphi_*std_radphi_, 0,
		0, 0, std_radrd_*std_radrd_;

	//move to measurement?
	rLaser_ << std_laspx_*std_laspx_, 0,
		0, std_laspy_*std_laspy_;

	MatrixXd X_Sigma_ = MatrixXd(n_x_, n_sigma_points_);
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
	//From video
	if (!is_initialized_) {
		time_us_ = meas_package.timestamp_;
		double matrixEntry = 0.5 / (n_aug_ + lambda_);
		for (int i = 0; i < n_sigma_points_; i++) {
			weights_(i) = matrixEntry;
		}
		weights_(0) = 2 * matrixEntry;

		//Initialize x_, P_, previous time, anything else needed
		P_ << 1, 0, 0, 0, 0,
			0, 1, 0, 0, 0,
			0, 0, 1, 0, 0,
			0, 0, 0, 1, 0,
			0, 0, 0, 0, 1;   //replace with identity matrix TODO
		if (meas_package.sensor_type == MeasurementPackage::LASER) {
			//Initialize here
			x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
		}
		else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			//Initialize Here
			float rho = meas_package.raw_measurements_[0];
			float phi = meas_package.raw_measurements_[1];
			float rho_dot = meas_package.raw_measurements_[2];
			float p_x = rho*cos(phi);
			float p_y = rho*sin(phi);
			x_ << p_x, p_y, rho_dot, 0, 0;  //need to be v instead of rho_dot?  difference?
		}
	is_initialized_ = true;
	return;									  
	}  //end of is_initialized

	double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
										
										
Prediction(delta_t);

if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
	UpdateLidar(meas_package);
}
 
else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
	UpdateRadar(meas_package);
}
time_us_ = meas_package.timestamp_;								   
}  //end of ProcessMeasurement




/**
* Predicts sigma points, the state, and the state covariance matrix.
* @param {double} delta_t the change in time (in seconds) between the last
* measurement and this one.
*/
void UKF::Prediction(double delta_t) {
	/**Complete this function! Estimate the object's location. Modify the state
	vector, x_. Predict sigma points, the state, and the state covariance matrix.
	*/
	//L7.18... create the sigma points!
	//create augmented mean state

	VectorXd x_aug_ = VectorXd(n_aug_);
	x_aug_.fill(0.0);
	x_aug_.head(5) = x_;
	x_aug_(5) = 0;
	x_aug_(6) = 0;

	//create augmented covariance matrix
	MatrixXd P_aug_ = MatrixXd(n_aug_, n_sigma_points_);
	P_aug_.fill(0.0);
	P_aug_.topLeftCorner(5, 5) = P_;
	P_aug_(5, 5) = std_a_*std_a_;
	P_aug_(6, 6) = std_yawdd_*std_yawdd_;

	//create square root matrix
	MatrixXd L = P_aug_.llt().matrixL();

	//create augmented sigma points
	MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sigma_points_);
	Xsig_aug.col(0) = x_aug_;
	for (int i = 0; i< n_aug_; i++) { //
		Xsig_aug.col(i + 1) = x_aug_ + sqrt(lambda_ + n_aug_) * L.col(i);//
		Xsig_aug.col(i + 1 + n_aug_) = x_aug_ - sqrt(lambda_ + n_aug_) * L.col(i);//
	}

	//L7.21... predict some things!
	for (int i = 0; i< 2 * n_aug_ + 1; i++)
	{
		//Define Values
		double p_x = Xsig_aug(0, i);  //maybe take the definitions and put in .h?
		double p_y = Xsig_aug(1, i);
		double v = Xsig_aug(2, i);
		double yaw = Xsig_aug(3, i);
		double yawd = Xsig_aug(4, i);
		double nu_a = Xsig_aug(5, i);
		double nu_yawdd = Xsig_aug(6, i);
		//predicted state values
		double px_p;
		double py_p;

		//avoid division by zero
		if (fabs(yawd) > 0.001) {
			px_p = p_x + v / yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
			py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd*delta_t));
		}
		else {
			px_p = p_x + v*delta_t*cos(yaw);
			py_p = p_y + v*delta_t*sin(yaw);
		}

		double v_p = v;
		double yaw_p = yaw + yawd*delta_t;
		double yawd_p = yawd;

		//add noise
		px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
		py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
		v_p = v_p + nu_a*delta_t;

		yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
		yawd_p = yawd_p + nu_yawdd*delta_t;

		//write predicted sigma point into right column
		Xsig_pred_(0, i) = px_p;
		Xsig_pred_(1, i) = py_p;
		Xsig_pred_(2, i) = v_p;
		Xsig_pred_(3, i) = yaw_p;
		Xsig_pred_(4, i) = yawd_p;
	}

	/*	//Predict Mean and Covariance  - L7.24
	// set weights
	double weight_0 = lambda_/(lambda_+n_aug);
	weights(0) = weight_0;
	for (int i=1; i<2*n_aug+1; i++) {  //2n+1 weights
	double weight = 0.5/(n_aug+lambda_);
	weights(i) = weight;
	*/
}

//predicted state mean
x_.fill(0.0);
for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
	x_ = x_ + weights_(i) * Xsig_pred.col(i);
}

//predicted state covariance matrix
P_.fill(0.0);
for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

										   // state difference
	VectorXd x_diff = Xsig_pred_.col(i) - x_;
	//angle normalization
	while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
	while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;

	P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
}


/**
* Updates the state and the state covariance matrix using a laser measurement.
* @param {MeasurementPackage} meas_package
*/
void UKF::UpdateLidar(MeasurementPackage meas_package) {
	//no reason to use any nonlinear techniques here!  Use regular kalman filter equations
	n_z_ = 2;
	MatrixXd z_ = Xsig_pred_.block(0, 0, n_z_, n_sigma_points_);
	VectorXd z_perd_ = VectorXd(n_z_);
	z_pred_.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		z_pred = z_pred + weights(i) * Zsig.col(i);
	}
	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z_, n_z_);
	S.fill(0.0);
	for (int i = 0; i < n_sig_; i++) {

		VectorXd zDiff = Zsig.col(i) - z_pred;

		while (zDiff(1)> M_PI) zDiff(1) -= 2.*M_PI;
		while (zDiff(1)<-M_PI) zDiff(1) += 2.*M_PI;



		S = S + weights_(i) * zDiff * zDiff.transpose();
	}

	S = S + rLaser_;

	
	MatrixXd CrossCorr = MatrixXd(n_x_, n_z_);
	
	CrossCorr.fill(0.0);
	for (int i = 0; i < n_sig_; i++) {

		//residual
		VectorXd  zDiff = Zsig.col(i) - z_pred;
		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) { // Radar




			while (zDiff(1)> M_PI) zDiff(1) -= 2.*M_PI;
			while (zDiff(1)<-M_PI) zDiff(1) += 2.*M_PI;




		}
		// State difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		// Angle normalization
		while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
		while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;
		CrossCorr = CrossCorr + weights_(i) * x_diff * zDiff.transpose();
	}
	// Measurements
	VectorXd z = meas_package.raw_measurements_;
	//Kalman gain K;
	MatrixXd K = CrossCorr * S.inverse();
	// Residual


	VectorXd zDiff = z - z_pred;
	if (meas_package.sensor_type_ == MeasurementPackage::RADAR) { // Radar
																  // Angle normalization
		while (zDiff(1)> M_PI) zDiff(3) -= 2.*M_PI;
		while (zDiff(3)<-M_PI) zDiff(3) += 2.*M_PI;
	}

	// Update state mean and covariance matrix
	x_ = x_ + K * zDiff;
	P_ = P_ - K * S * K.transpose();
	// Calculate NIS
	NIS_laser_ = z.transpose() * S.inverse() * z;


}

/**
* Updates the state and the state covariance matrix using a radar measurement.
* @param {MeasurementPackage} meas_package
*/
void UKF::UpdateRadar(MeasurementPackage meas_package) {
	/**Use radar data to update the belief about the object's
	position. Modify the state vector, x_, and covariance, P_.
	int n_z_ = 3;


	Also calculates the radar NIS.*/
	//L7.27
	//transform sigma points into measurement space
	for (int i = 0; i < 2 * n_aug_+ 1; i++) {  //2n+1 simga points

											   // extract values for better readibility
		double p_x = Xsig_pred(0, i);
		double p_y = Xsig_pred(1, i);
		double v = Xsig_pred(2, i);
		double yaw = Xsig_pred(3, i);

		double v1 = cos(yaw)*v;
		double v2 = sin(yaw)*v;

		// measurement model
		Zsig(0, i) = sqrt(p_x*p_x + p_y*p_y);
		Zsig(1, i) = atan2(p_y, p_x);
		Zsig(2, i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);
	}

	//mean predicted measurement
	VectorXd z_pred_ = VectorXd(n_z);
	z_pred_.fill(0.0);

	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		z_pred_ = z_pred_ + weights(i) * Zsig.col(i);  //not sure if this is right 
	}

	MatrixXd S = MatrixXd(n_z, n_z);
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
											   //residual
		VectorXd zDiff = Zsig.col(i) - z_pred;

		//angle normalization
		while (zDiff(1)> M_PI) zDiff(1) -= 2.*M_PI;
		while (zDiff(1)<-M_PI) zDiff(1) += 2.*M_PI;


		S = S + weights(i) * zDiff * zDiff.transpose();
	}

	S = S + rRadar_;

	//measurement covariance matrix S

	/*Update Radar*/
	//L7.30
	VectorXd z = meas_package.raw_measurements_;
	CrossCorr.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

											   //residual
		VectorXd zDiff = Zsig.col(i) - z_pred;
		//angle normalization
		while (zDiff(1)> M_PI) zDiff(1) -= 2.*M_PI;
		while (zDiff(1)<-M_PI) zDiff(1) += 2.*M_PI;

		// state difference
		VectorXd x_diff = Xsig_pred.col(i) - x_;
		//angle normalization
		while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
		while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;

		CrossCorr = CrossCorr + weights(i) * x_diff * zDiff.transpose();
	}



	//Gain Matrix
	MatrixXd K = CrossCorr * S.inverse();


	VectorXd zDiff = z - z_pred_;

	//angle normalization
	while (zDiff(1)> M_PI) zDiff(1) -= 2.*M_PI;
	while (zDiff(1)<-M_PI) zDiff(1) += 2.*M_PI;

	//update state mean and covariance matrix
	x_ = x_ + K*zDiff;
	P_ = P_ - K*S*K.transpose();
	NIS_radar_ = z.transpose() * S.inverse() * z;
}