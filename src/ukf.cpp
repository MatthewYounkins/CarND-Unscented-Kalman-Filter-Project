#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

//Unscented Kalman filter
UKF::UKF() {						// Initialize UKF
	is_initialized_ = false;		// This will change after the system is initialized
	time_us_ 	= 0.0;				// System time, in microseconds

	std_a_ 		= 1.0;  			// Potential deviation in longitudinal acceleration in m/s^2
	std_yawdd_ 	= 0.5;  			// Potential deviation in yaw acceleration in rad/s^2

	//Some example iterations of the above:
	//     std_a_  std_yawdd	X		Y		VX		VY
	//Using 3.0  	1.0 	>> 	0.0732 	0.0853 	0.3573 	0.2441
	//Using 2.0		1.0		>>	0.0700	0.0838	0.3433	0.2291
	//Using 1.0		1.0		>>	0.0647	0.0836	0.3350	0.2193
	//Using 1.0		0.5		>>	0.0661	0.0827	0.3323	0.2146

	//1.0 m/s^2 is rather slow, but wouldn't you use a somewhat adaptive bound here anyway?

	std_laspx_ 	= 0.15;				// Standard dev of laser position, m
	std_laspy_ 	= 0.15;				// Standard dev of laser position, m
	std_radr_ 	= 0.3;				// Standard dev of radar measurement distance in m
	std_radphi_ = 0.03;				// Standard dev of angle measurement in rad
	std_radrd_ 	= 0.3;				// Standard dev in rate of change of radius in m/s

	n_x_ 		= 5; 				// Number of measurements
	n_aug_ 		= n_x_ + 2;			// Number of positions in the augmented states
	nSigmaPts_	= 2 * n_aug_ + 1;	// How many sigma points will we use?
	lambda_ 	= 3 - n_aug_;		// Sigma Spread
		
	xSigmaPred_ = MatrixXd(n_x_, nSigmaPts_);	// Matrix for sigma points
	w_ 			= VectorXd(nSigmaPts_);			// Vector for weights
	x_ 			= VectorXd(n_x_);  				// [pos1 pos2 vel_abs yaw_angle yaw_rate] = 5 measurements in mks and radians
	P_			= MatrixXd(n_x_, n_x_);			// Initialize the P matrix
	rRadar_ 	= MatrixXd(3, 3);				// Covariance for Radar 
	rLidar_ 	= MatrixXd(2, 2);				// Covariance for Lidar	
    
	// Measurement noise covariance matrices initialization
	rRadar_	<< 	std_radr_*std_radr_,	0,							0,
				0,						std_radphi_*std_radphi_, 	0,
				0,						0,							std_radrd_*std_radrd_;
	rLidar_	<< 	std_laspx_*std_laspx_,	0,
				0,						std_laspy_*std_laspy_;
}

UKF::~UKF() {}

// Replace measurement stuff
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
	if (!is_initialized_) {
		time_us_ = meas_package.timestamp_;
		P_.topLeftCorner(5,5).setIdentity(); 	// Initial covariance matrix
		if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
			x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
			}
		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			float rho = meas_package.raw_measurements_[0];
			float phi = meas_package.raw_measurements_[1];
			float rho_dot = meas_package.raw_measurements_[2];
			float p_x = rho * cos(phi); 
			float p_y = rho * sin(phi);
			x_ << p_x, p_y, rho_dot, 0, 0;				//rho dot is not really the same thing as v, although it is in this simple case.
		}
    	float vectorEntry = 0.5 / (lambda_ + n_aug_); 	//save some calc time
		w_(0) = 2*lambda_*vectorEntry;				    //set weights
		for (int i = 1; i < w_.size(); i++) w_(i) = vectorEntry;
		is_initialized_ = true;
		return;
		} //end of initialization
	
	Prediction((meas_package.timestamp_ - time_us_)*1e-6);
	time_us_ = meas_package.timestamp_;
	if (meas_package.sensor_type_ == MeasurementPackage::RADAR) UpdateRadar(meas_package);
	if (meas_package.sensor_type_ == MeasurementPackage::LASER) UpdateLidar(meas_package);
}	//end of ProcessMeasurement Function

// Sigma points, state, and covariance matrix.

void UKF::Prediction(double delta_t_) {			// For reference:  L7.18
	VectorXd x_aug = VectorXd(n_aug_);			// Augmented mean vector
	MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);	// Augmented state covarience matrix
	MatrixXd Xsig_aug = MatrixXd(n_aug_, nSigmaPts_); // Sigma point matrix
	
	x_aug.fill(0.0);
	x_aug.head(n_x_) = x_;						//don't need to set 6 to zero
	
	P_aug.fill(0);
	P_aug.topLeftCorner(n_x_,n_x_) = P_;
	P_aug(5,5) = std_a_*std_a_;
	P_aug(6,6) = std_yawdd_*std_yawdd_;
	
	MatrixXd L = P_aug.llt().matrixL();			//square root matrix
	Xsig_aug.col(0) = x_aug;					// Create sigma points
	
	for(int i = 0; i < n_aug_; i++) {
		Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
		Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
	}

	for (int i = 0; i< nSigmaPts_; i++){			//Reference:  L7.21
		const double p_x 		= Xsig_aug(0,i);	//Position, X direction
		const double p_y 		= Xsig_aug(1,i);	//Position, Y direction
		const double v 			= Xsig_aug(2,i);	//Velocity
		const double yaw 		= Xsig_aug(3,i);	//Yaw Angle
		const double yawd 		= Xsig_aug(4,i);	
		const double nu_a 		= Xsig_aug(5,i);
		const double nu_yawdd 	= Xsig_aug(6,i);
		
		// Predicted state values
		double px_p, py_p;
		
		if (fabs(yawd) > .0001) {    // Avoid division by zero	
			px_p = p_x + v/yawd * (sin(yaw + yawd*delta_t_) - sin(yaw));	//could precalc
			py_p = p_y + v/yawd * (cos(yaw) - cos(yaw + yawd*delta_t_) );	//could precalc
		}
		else {
			px_p = p_x + v*delta_t_*cos(yaw);
			py_p = p_y + v*delta_t_*sin(yaw);
		}
		
		double v_p 		= v;
		double yaw_p 	= yaw + yawd*delta_t_;
		double yawd_p 	= yawd;

		//add noise
		px_p 	= px_p + 0.5  * nu_a * delta_t_ * delta_t_ * cos(yaw);
		py_p 	= py_p + 0.5  * nu_a * delta_t_ * delta_t_ * sin(yaw);
		v_p  	= v_p + nu_a * delta_t_;
		yaw_p 	= yaw_p + 0.5 * nu_yawdd*delta_t_*delta_t_;
		yawd_p 	= yawd_p + nu_yawdd*delta_t_;

		//write predicted sigma point into right column
		xSigmaPred_(0,i) = px_p;
		xSigmaPred_(1,i) = py_p;
		xSigmaPred_(2,i) = v_p;
		xSigmaPred_(3,i) = yaw_p;
		xSigmaPred_(4,i) = yawd_p;
	}

	x_.fill(0.0);					//predicted state mean, could potentially make one line
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
		x_ = x_+ w_(i) * xSigmaPred_.col(i);
	}
	
	P_.fill(0.0);							// Predicted state covariance matrix
	for (int i = 0; i < nSigmaPts_; i++) {  //iterate over sigma points
		// State difference
		VectorXd xDiff_ = xSigmaPred_.col(i) - x_;
		// Angle normalization
		restrictAngle(&(xDiff_(3)));
		P_ = P_ + w_(i) * xDiff_ * xDiff_.transpose() ;
	}
} //End of Prediction Function

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
	//Could use regular Kalman filter equations for the linear model.  Instead this is UKF.
	//Perhaps I could speed things up by using the regular Kalman filer?
	int nLidar_ = 2; 
	MatrixXd z_ = xSigmaPred_.block(0, 0, nLidar_, nSigmaPts_);
	VectorXd z_pred_ = VectorXd(nLidar_);
	z_pred_.fill(0.0);
	for (int i = 0; i < nSigmaPts_; i++) {
		z_pred_ = z_pred_ + w_(i) * z_.col(i);
	}
	//measurement covariance matrix S
	MatrixXd S = MatrixXd(nLidar_, nLidar_);
	S.fill(0.0);
	for (int i = 0; i < nSigmaPts_; i++) {
		VectorXd zDiff_ = z_.col(i) - z_pred_;
		restrictAngle(&(zDiff_(1)));
		S = S + w_(i) * zDiff_ * zDiff_.transpose();
	}
	S = S + rLidar_;
	MatrixXd CrossCorr = MatrixXd(n_x_, nLidar_);
	CrossCorr.fill(0.0);
	for (int i = 0; i < nSigmaPts_; i++) {
		//residual
		VectorXd  zDiff_ = z_.col(i) - z_pred_;
		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) { // Radar
			restrictAngle(&(zDiff_(1)));
		}
		
		VectorXd xDiff_ = xSigmaPred_.col(i) - x_;					// State difference
		CrossCorr = CrossCorr + w_(i) * xDiff_ * zDiff_.transpose();
	}
	
	VectorXd z = meas_package.raw_measurements_;	//measured data
	MatrixXd K = CrossCorr * S.inverse();			//K is the Kalman gain
	VectorXd zDiff_ = z - z_pred_;
	if (meas_package.sensor_type_ == MeasurementPackage::RADAR) { // Radar
		restrictAngle(&(zDiff_(3)));
	}

	x_ = x_ + K * zDiff_;							//update mean value
	P_ = P_ - K * S * K.transpose();				//update covariance
	NIS_laser_ = z.transpose() * S.inverse() * z;	//calc NIS
}//End of UpdateLidar Function

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  	int nRadar_ = 3;								//r, phi, rdot
  	MatrixXd z_ = MatrixXd(nRadar_, nSigmaPts_);   	//sigma points
  	// Mean predicted measurement
  	VectorXd z_pred = VectorXd(nRadar_);
  	
	for (int i = 0; i < nSigmaPts_; i++) {
    	double p_x 	= xSigmaPred_(0,i);
    	double p_y 	= xSigmaPred_(1,i);
    	double v  	= xSigmaPred_(2,i);
    	double yaw 	= xSigmaPred_(3,i);
    	double v1 	= cos(yaw)*v;
    	double v2 	= sin(yaw)*v;
    	
    	z_(0,i) = sqrt(p_x*p_x + p_y*p_y);			//r
    	z_(1,i) = atan2(p_y,p_x);                   //phi.  Should check for p_x and p_y = 0; atan2(0,0) is undefined.
    	z_(2,i) = (p_x*v1 + p_y*v2 ) / z_(0,i);   	//r_dot
  	}

  	z_pred  = z_ * w_;
  	MatrixXd S = MatrixXd(nRadar_, nRadar_);		//Covariance Matrix
  	S.fill(0.0);									//initializing covariance
  	for (int i = 0; i < nSigmaPts_; i++) { 
    	VectorXd zDiff_ = z_.col(i) - z_pred;		//Calculate residual
    	restrictAngle(&(zDiff_(1)));
    	S = S + w_(i) * zDiff_ * zDiff_.transpose();
  	}
  	
  	S = S + rRadar_;
  
   	MatrixXd CrossCorr = MatrixXd(n_x_, nRadar_);
  	CrossCorr.fill(0.0);							//Initialize cross-corelation
  	for (int i = 0; i < nSigmaPts_; i++) { 
    	VectorXd zDiff_ = z_.col(i) - z_pred;
    	restrictAngle(&(zDiff_(1)));				//This is required - taking it out makes the system fail
		VectorXd xDiff_ = xSigmaPred_.col(i) - x_;
		CrossCorr = CrossCorr + w_(i) * xDiff_ * zDiff_.transpose();
  	}
 	VectorXd z = meas_package.raw_measurements_;	//measured data
	MatrixXd K = CrossCorr * S.inverse();			//K is the Kalman gain
	VectorXd zDiff_ = z - z_pred;
  	restrictAngle(&(zDiff_(1)));
  	
  	x_ = x_ + K * zDiff_;							//update mean value
  	P_ = P_ - K * S * K.transpose();				//update covariance
  	NIS_radar_ = z.transpose() * S.inverse() * z;	//calc NIS
}//End of UpdateRadar Function



void UKF::restrictAngle(double *angle) {			//restricts angle to -pi and pi.
    while (*angle > M_PI)  *angle -= 2. * M_PI;
    while (*angle < -M_PI) *angle += 2. * M_PI;
}

