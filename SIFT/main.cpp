#include<iostream>
#include<opencv2\opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include<opencv2\core\core.hpp>
#include<opencv2\highgui.hpp>
#include<opencv2\xfeatures2d\nonfree.hpp>
#include<opencv2\features2d.hpp>
#include <algorithm>
#include <tuple>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

class Sift_feature
{
public:

	Mat generate_base_img(Mat scr) {
		/*
		generate base image
		inputs: 
		 - source image of size (r, c).
		outputs:
		 - blurred image of size (2*r, 2*c).
		*/
		GaussianBlur(scr, scr, Size(0, 0), 0.5, 0.5);
		resize(scr, scr, Size(0, 0), 2, 2, INTER_LINEAR);
		return scr;
	}

	//-------------------------------------------------------------------------------

	int compute_Number_Of_Octaves(Size image_shape) {
		/*
		compute number of octaves
		inputs:
		- size of the image.
		outputs:
		- number of octaves (integer number).
		*/
		return int(round(log(min(image_shape.height, image_shape.width)) / log(2) - 1));
	}

	//-------------------------------------------------------------------------------

	vector<double> generate_sigmas(double sigma, int num_intervals) {
		/*
		generate sigma values for blurring 
		inputs:
		- sigma value.
		- number of intervals
		outputs:
		- vector of double values.
		*/
		double new_sigma;
		int num_images_per_octave = num_intervals + 3;
		double k = pow(2, (1. / num_intervals));
		vector<double> sigmas;
		sigmas.push_back(sigma);

		for (int i = 1; i < num_images_per_octave; i++)
		{
			new_sigma = pow(k, i) * sigma;
			sigmas.push_back(new_sigma);
		}

		return sigmas;
	}

	//-------------------------------------------------------------------------------

	tuple<vector<vector<Mat>>, vector<vector<Mat>>> Create_Scale_Space(Mat scr, int num_intervals, vector<double> sigmas) {
		/*
		create scale space and difference of gaussian images
		inputs:
		- input image.
		- number of intervals.
		- vector of sigma values.
		outputs:
		- tuple of scale space and difference of gaussian images.
		*/
		int num_octaves = compute_Number_Of_Octaves(scr.size());
		// define vector of vectors of mats for both scale space and DoG images.
		vector<vector<Mat>> scale_space;
		vector<vector<Mat>> DoG;
		// initialize both scale space and DoG
		for (int oct = 0; oct < num_octaves; oct++)
		{
			scale_space.push_back(vector<Mat>(num_intervals + 3));
			DoG.push_back(vector<Mat>(num_intervals + 2));
		}
		// blur the base image with first sigma value 
		GaussianBlur(scr, scr, Size(0, 0), sigmas[0], sigmas[0]);
		// copy the blurred image to the first octave and first scale image
		scr.copyTo(scale_space[0][0]);
		// two for loops for compute scale images and DoG images in all octaves
		for (int oct = 0; oct < num_octaves; oct++)
		{

			for (int scale = 1; scale < num_intervals + 3; scale++)
			{
				GaussianBlur(scale_space[oct][scale - 1], scale_space[oct][scale], Size(0, 0), sigmas[scale], sigmas[scale]);
				DoG[oct][scale - 1] = scale_space[oct][scale] - scale_space[oct][scale - 1];
			}
			
			// downsampling until reach the final octave
			if (oct < num_octaves - 1)
			{
				resize(scale_space[oct][0], scale_space[oct + 1][0], Size(0, 0), 0.5, 0.5, INTER_LINEAR);
			}

		}
		return make_tuple(scale_space, DoG);
	}

	//-------------------------------------------------------------------------------

	vector<KeyPoint> Extrema(vector<vector<Mat>> DoG, vector<vector<Mat>> scale_images, int num_intervals, double sigma) {
		tuple<Point, int> candidate;
		double threshold = floor(0.5 * 0.04 / num_intervals * 255); // contrast_threshold=0.04
		vector<KeyPoint> keypoints;
		for (int oct = 0; oct < DoG.size(); oct++)
		{
			for (int scale = 1; scale < DoG[0].size() - 1; scale++)
			{
				for (int i = 5; i < DoG[oct][0].size().height - 5; i++)
				{
					for (int j = 5; j < DoG[oct][0].size().width - 5; j++)
					{
						if (is_it_extremum(DoG[oct][scale](Rect(j - 1, i - 1, 3, 3)), 
							DoG[oct][scale + 1](Rect(j - 1, i - 1, 3, 3)), 
							DoG[oct][scale - 1](Rect(j - 1, i - 1, 3, 3)), threshold)) {

							candidate = localize_Extremum(i, j, scale, DoG[oct], oct);
							if (get<0>(candidate).x != -1 && get<0>(candidate).y != -1)
							{
								vector<float> Orientations = compute_orientation(get<0>(candidate), oct, scale, scale_images[oct][get<1>(candidate)]);
								
								for (int angle = 0; angle < Orientations.size(); angle++)
								{
									KeyPoint key;
									key.angle = Orientations[angle];
									key.pt = get<0>(candidate);
									key.octave = oct;
									key.size = get<1>(candidate);
									keypoints.push_back(key);
								}
								
							}
						}

					}
				}
				
			}
		}
		vector<KeyPoint> unique_keys;
		unique_keys = remove_duplicate(keypoints);
		return unique_keys;
	}

	//-------------------------------------------------------------------------------

	bool is_it_extremum(Mat current_k, Mat k_up, Mat k_down, double threshold) {
		/*
		check if the center pixel is extremum among its 26 neighbors.
		inputs:
		- current kernel.
		- kernel up.
		- kernel down.
		- threshold value.
		outputs:
		- boolean value indicates if the center pixel is extremum or not.
		*/
		float center_pixel = current_k.at<float>(1, 1);
		if (abs(center_pixel) > threshold)
		{
			MatConstIterator_<float> it_curr = current_k.begin<float>(), it_curr_end = current_k.end<float>();
			MatConstIterator_<float> it_curr_before_center = next(current_k.begin<float>(), 4);
			MatConstIterator_<float> it_curr_after_center = next(current_k.begin<float>(), 5);

			MatConstIterator_<float> it_up = k_up.begin<float>(), it_up_end = k_up.end<float>();

			MatConstIterator_<float> it_down = k_down.begin<float>(), it_down_end = k_down.end<float>();

			if (all_of(it_up, it_up_end, [center_pixel](float i) {return center_pixel > i; })
				&& all_of(it_down, it_down_end, [center_pixel](float i) {return center_pixel > i; })
				&& all_of(it_curr, it_curr_before_center, [center_pixel](float i) { return center_pixel > i; })
				&& all_of(it_curr_after_center, it_curr_end, [center_pixel](float i) {return center_pixel > i; }))
			{
				return true;
			}
			else if (all_of(it_up, it_up_end, [center_pixel](float i) {return center_pixel < i; })
				&& all_of(it_down, it_down_end, [center_pixel](float i) {return center_pixel < i; })
				&& all_of(it_curr, it_curr_before_center, [center_pixel](float i) {return center_pixel < i; })
				&& all_of(it_curr_after_center, it_curr_end, [center_pixel](float i) {return center_pixel < i; }))
			{
				return true;
			}

		}
		return false;
	}
	//-------------------------------------------------------------------------------
	
	

	tuple<Point, int> localize_Extremum(int ii, int jj, int sc, vector<Mat> Octave, int octave_index) {
		/*
		obtain the interpolated estimate of the location of the extremum.
		inputs:
		- coordinates of the extremum (i, j, scale). 
		- octave images. 
		- octave index.
		outputs:
		- actual extremum point and the corresponding scale.
		*/
		Point P;
		Mat gradient, Hessian, extremum_update;
		Mat current_img, img_up, img_down;
		int previous_scale = -1;
		int attempt;
		for (attempt = 0; attempt < 5; attempt++)
		{
			Octave[sc].copyTo(current_img);
			Octave[sc + 1].copyTo(img_up);
			Octave[sc - 1].copyTo(img_down);
			// normalize images into range [0-1]
			if (previous_scale != sc)
			{
				previous_scale = sc;
				normalize(current_img, current_img, 0, 1, NORM_MINMAX);
				normalize(img_up, img_up, 0, 1, NORM_MINMAX);
				normalize(img_down, img_down, 0, 1, NORM_MINMAX);
			}
			// compute gradient 
			gradient = compute_gradient(current_img(Rect(jj - 1, ii - 1, 3, 3)),
				img_up(Rect(jj - 1, ii - 1, 3, 3)), img_down(Rect(jj - 1, ii - 1, 3, 3)));
			// compute Hessian matrix
			Hessian = compute_Hessian(current_img(Rect(jj - 1, ii - 1, 3, 3)),
				img_up(Rect(jj - 1, ii - 1, 3, 3)), img_down(Rect(jj - 1, ii - 1, 3, 3)));
			// compute the location of the extremum
			solve(Hessian, gradient, extremum_update);
			extremum_update = - extremum_update;
			// stop if the offset is less than 0.5 in all of its three dimensions
			if (abs(extremum_update.at<float>(0)) < 0.5 && abs(extremum_update.at<float>(1)) < 0.5 && abs(extremum_update.at<float>(2)) < 0.5)
			{
				break;
			}
			// update coordinates
			ii += int(round(extremum_update.at<float>(1)));
			jj += int(round(extremum_update.at<float>(0)));
			sc += int(round(extremum_update.at<float>(2)));
			//check if extremum is outside the image
			if (ii < 5 || ii > current_img.size().height - 5 || jj < 5 || jj > current_img.size().width - 5 || sc < 1 || sc >= Octave.size() - 1)
			{
				P.x = -1;
				P.y = -1;
				return make_tuple(P, -1);
			}
		}
		if (attempt >= 4)
		{
			P.x = -1;
			P.y = -1;
			return make_tuple(P, -1);
		}
		// elemnating low contrast
		if (abs(current_img.at<float>(ii, jj) < 0.03)) // CONTRAST_THRESHOLD = 0.03
		{
			P.x = -1;
			P.y = -1;
			return make_tuple(P, -1);
		}
		// eleminating edges
		double trace = Hessian.at<float>(0, 0) + Hessian.at<float>(1, 1);
		double deter = Hessian.at<float>(0, 0) * Hessian.at<float>(1, 1) - Hessian.at<float>(0, 1) * Hessian.at<float>(1, 0);
		double curvature = (trace*trace / deter);
		if (deter < 0 || curvature > 10)   // curv_threshold = 10
		{
			P.x = -1;
			P.y = -1;
			return make_tuple(P, -1);
		}
		P.x = jj * pow(2, octave_index);
		P.y = ii * pow(2, octave_index);
		
		return make_tuple(P, sc);
	}

	//-------------------------------------------------------------------------------
	Mat compute_gradient(Mat current, Mat up, Mat down) {
		double dx, dy, dsigma;
		dx = 0.5 * (current.at<float>(1, 2) - current.at<float>(1, 0));
		dy = 0.5 * (current.at<float>(2, 1) - current.at<float>(0, 1));
		dsigma = 0.5 * (up.at<float>(1, 1) - down.at<float>(1, 1));
		Mat gradient = (Mat_<float>(3,1) << dx, dy, dsigma);

		return gradient;
	}

	//-------------------------------------------------------------------------------

	Mat compute_Hessian(Mat current, Mat up, Mat down) {
		double dxx, dyy, dss, dxy, dxs, dys;
		dxx = current.at<float>(1, 2) - 2 * current.at<float>(1, 1) + current.at<float>(1, 0);
		dyy = current.at<float>(2, 1) - 2 * current.at<float>(1, 1) + current.at<float>(0, 1);
		dss = up.at<float>(1, 1) - 2 * current.at<float>(1, 1) + down.at<float>(1, 1);
		dxy = 0.25 * (current.at<float>(2, 2) - current.at<float>(0, 2) - current.at<float>(2, 0) + current.at<float>(0, 0));
		dxs = 0.25 * (up.at<float>(1, 2) - down.at<float>(1, 2) - up.at<float>(1, 0) + down.at<float>(1, 0));
		dys = 0.25 * (up.at<float>(2, 1) - down.at<float>(2, 1) - up.at<float>(0, 1) + down.at<float>(0, 1));
		Mat Hessian = (Mat_<float>(3, 3) << dxx, dxy, dxs,
			dxy, dyy, dys,
			dxs, dys, dss);
		return Hessian;
	}


	//-------------------------------------------------------------------------------

	vector<float> compute_orientation(Point P, int octave, int scale, Mat gaussian_image) {

		double sigma = scale * 1.5;
		vector<float> Orientations;
		Mat kernel = gaussian_kernel(sigma);
		int radius = int(2 * ceil(sigma) + 1);
		int x, y;
		double weight;
		Mat raw_histogram = Mat::zeros(36, 1, CV_64FC1); // num_bins  = 36 


		for (int i = -radius; i <= radius; i++)
		{
			y = int(round((P.y / pow(2, octave)))) + i;
			if (y <= 0 || y >= gaussian_image.rows - 1) continue;

			for (int j = -radius; j <= radius; j++)
			{
				x = int(round((P.x / pow(2, octave)))) + j;
				if (x <= 0 || x >= gaussian_image.cols - 1) continue;

				double dx = gaussian_image.at<float>(y, x + 1) - gaussian_image.at<float>(y, x - 1);
				double dy = gaussian_image.at<float>(y + 1, x) - gaussian_image.at<float>(y - 1, x);
				double magnitude = sqrt(dx * dx + dy * dy);
				double orientation = atan2(dy, dx)  * (180.0 / 3.14159265);

				if (orientation < 0) orientation = orientation + 360;

				int histogram_index = int(floor(orientation * 36.0 / 360.0));
				/*Each sample added to the histogram is weighted by its gradient magnitude
				and by a Gaussian-weighted circular window with a  that is 1.5 times that of the scale
				of the keypoint.*/
				weight = kernel.at<double>(j + radius, i + radius) * magnitude;
				raw_histogram.at<double>(histogram_index, 0) += weight;

			}
		}

		/*Finally, a parabola is fit to the 3 histogram values closest to each peak to interpolate the peak position
		for better accuracy.
		You can find more details from here: https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
		*/

		double maxVal;
		Point maxLoc;
		double left_value, right_value, orientation, interpolated_peak_index;
		minMaxLoc(raw_histogram, NULL, &maxVal, NULL, &maxLoc);

		for (int bin = 0; bin < raw_histogram.rows; bin++)
		{
			if (raw_histogram.at<double>(bin, 0) >= (0.8 * maxVal))
			{
				if (bin == 0) left_value = raw_histogram.at<double>(35, 0);
				else left_value = raw_histogram.at<double>(bin - 1, 0);

				if (bin == 35) right_value = raw_histogram.at<double>(0, 0);
				else right_value = raw_histogram.at<double>(bin + 1, 0);

				interpolated_peak_index = bin + 0.5 * (left_value - right_value) / (left_value - 2 * raw_histogram.at<double>(bin, 0) + right_value);
				orientation = interpolated_peak_index * 360.0 / 36.0;

				if (orientation < 0 || orientation >= 360) orientation = abs(360 - abs(orientation));
				Orientations.push_back(orientation);
			}
		}
		return Orientations;

	}


	//-------------------------------------------------------------------------------
	vector<KeyPoint> remove_duplicate(vector<KeyPoint> keypoints) {
		vector<KeyPoint> unique_keypoints;

		if (keypoints.size() < 2)
		{
			return keypoints;
		}

		struct myclass {
			bool operator() (KeyPoint key1, KeyPoint key2) {
				if (key1.pt.x != key2.pt.x)
				{
					return (key1.pt.x < key2.pt.x);
				}
				else
				{
					return (key1.pt.y < key2.pt.y);
				}
			}
		} myobject;
		sort(keypoints.begin(), keypoints.end(), myobject);

		unique_keypoints.push_back(keypoints[0]);
		for (int i = 1; i < keypoints.size(); i++)
		{
			if (unique_keypoints.back().pt.x != keypoints[i].pt.x || unique_keypoints.back().pt.y != keypoints[i].pt.y || unique_keypoints.back().angle != keypoints[i].angle)
			{
				unique_keypoints.push_back(keypoints[i]);
			}
		}

		return unique_keypoints;
	}
	
	//-------------------------------------------------------------------------------

	Mat gaussian_kernel(double sigma) {
		/* https://theailearner.com/2019/05/06/gaussian-blurring/
		For Gaussian, we know that 99.3% of the distribution falls within 3 standard deviations after 
		which the values are effectively close to zero. So, we limit the kernel size to contain only 
		values within 3? from the mean. This approximation generally yields a result sufficiently 
		close to that obtained by the entire Gaussian distribution.*/

		/*Note: The approximated kernel weights would not sum exactly 1 so, normalize the weights 
		by the overall kernel sum. Otherwise, this will cause darkening or brightening of the image.
		A normalized 3×3 Gaussian filter is shown below (See the weight distribution)*/

		/*First, the Gaussian kernel is linearly separable. This means we can break 
		any 2-d filter into two 1-d filters. Because of this, the computational complexity 
		is reduced from O(n2) to O(n). Let’s see an example*/
		int r = ceil(3 * sigma);
		Mat kernel(2 * r + 1, 2 * r + 1, CV_64FC1);
		for (int i = -r; i <= r; i++)
		{
			for (int j = -r; j <= r; j++)
			{
				kernel.at<double>(i + r, j + r) = exp(-(i*i + j*j) / (2.0 * sigma*sigma));
			}
		}
		kernel = kernel / sum(kernel);
		return kernel;
	}

	//-------------------------------------------------------------------------------

	vector<Mat> descriptor(vector<KeyPoint> keypoints, vector<vector<Mat>> scale_space) {
		int w = 16;
		Mat kernel = gaussian_kernel(w / 6.0);
		Mat feature_vector = Mat::zeros(128, 1, CV_64FC1);
		vector<Mat> features;
		for (int kp = 0; kp < keypoints.size(); kp++)
		{
			int octave = keypoints[kp].octave;
			int scale = keypoints[kp].size;
			int y_c = keypoints[kp].pt.y / pow(2, octave);
			int x_c = keypoints[kp].pt.x / pow(2, octave);

			Mat magnitude = Mat::zeros(Size(17, 17), CV_64FC1);
			Mat orientation = Mat::zeros(Size(17, 17), CV_64FC1);

			Mat gaussian_image = scale_space[octave][scale];
			if (x_c - w/2 >0 && y_c - w/2 > 0 && x_c + w/2 < gaussian_image.cols -1 && y_c + w/2 < gaussian_image.rows -1)
			{
				for (int i = -w / 2; i <= w / 2; i++)
				{
					int y = y_c + i;
					for (int j = -w / 2; j <= w / 2; j++)
					{
						int x = x_c + j;

						double dx = gaussian_image.at<float>(y, x + 1) - gaussian_image.at<float>(y, x - 1);
						double dy = gaussian_image.at<float>(y + 1, x) - gaussian_image.at<float>(y - 1, x);
						magnitude.at<double>(i + w/2, j + w/2) = sqrt(dx * dx + dy * dy);
						double theta = atan2(dy, dx)  * (180.0 / 3.14159265);
						if (theta < 0) theta = theta + 360;
						orientation.at<double>(i + w/2, j + w/2) = theta;
						
					}
				}
				Mat weighted_grad = magnitude.mul(kernel);
				Mat Q, hist_Q;
				for (int i = 0; i <=13 ; i = i+4)
				{
					int m = 0;
					for (int j = 0; j <=13; j = j+4)
					{
						Q = orientation(Rect(i, j, 4, 4));
						hist_Q = compute_hist(Q);
						hist_Q.copyTo(feature_vector(Rect(0, m, 1, 8)));
						m = m + 8;
						if (j == 4) j = j + 1;
					}
					if (i == 4) i = i + 1;
				}
				feature_vector = feature_vector / max(1e-6, norm(feature_vector, NORM_L2));
				threshold(feature_vector, feature_vector, 0.2, 255,THRESH_TRUNC);
				feature_vector = feature_vector / max(1e-6, norm(feature_vector, NORM_L2));
				features.push_back(feature_vector);
			}
		}
		return features;
	}

	//-------------------------------------------------------------------------------
	
	Mat compute_hist(Mat scr) {
		Mat hist = Mat::zeros(8, 1, CV_64FC1);
		int value = 0;
		int quantize_value;
		for (int i = 0; i < scr.rows; i++)
		{
			for (int j = 0; j < scr.cols; j++)
			{
				value = scr.at<double>(i, j);
				quantize_value = quantize_orientation(value);
				hist.at<double>(quantize_value) = hist.at<double>(quantize_value) + 1;
			}
		}
		return hist;
	}

	//-------------------------------------------------------------------------------

	int quantize_orientation(double angle) {
		return floor(angle / 45.0);
	}

	//-------------------------------------------------------------------------------
};




void main() {
	
	Mat img, scr, base_img;

	img = imread("cameraman.tif", 0);
	img.convertTo(scr, CV_32FC1);

	int num_intervals = 3;
	double sigma = 1.6;
	Sift_feature mysift;
	base_img = mysift.generate_base_img(scr);
	
	vector<double> sigmas = mysift.generate_sigmas(sigma, num_intervals);
	tuple<vector<vector<Mat>>, vector<vector<Mat>>> Scale_DoG = mysift.Create_Scale_Space(base_img, num_intervals, sigmas);
	vector<KeyPoint> keypoints = mysift.Extrema(get<1>(Scale_DoG), get<0>(Scale_DoG), num_intervals, sigma);
	vector<Mat> key_descriptors = mysift.descriptor(keypoints, get<0>(Scale_DoG));

	system("pause");
	waitKey(0);
}



/*
Mat img, output, descriptor;
img = imread("airship.jpg");

std::vector<KeyPoint> keypoints;

Ptr<SIFT> sift = SIFT::create();

sift->detect(img, keypoints,Mat());
//sift->detectAndCompute

drawKeypoints(img, keypoints, output);
imshow("sift_result.jpg", output);
waitKey(0);
*/
