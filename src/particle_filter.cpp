/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	this->num_particles = 100;
	default_random_engine gen;

	// Define normal distributions for sensor noise
	normal_distribution<double> std_x(x, std[0]);
	normal_distribution<double> std_y(y, std[1]);
	normal_distribution<double> std_theta(theta, std[2]);

	// Initialize all particles
	for(unsigned int i = 0; i < this->num_particles; i++)
	{
		Particle particle;
		particle.id = i;
		particle.x = std_x(gen);
		particle.y = std_y(gen);
		particle.theta = std_theta(gen);
		particle.weight = 1.0;
		this->particles.push_back(particle);
		this->weights.push_back(1.0);
	}
	this->is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;
	double pred_x;
	double pred_y;
	double pred_theta;

	for(unsigned int i = 0; i < this->num_particles; i++)
	{
		if(fabs(yaw_rate) < 0.0001)
		{
			// predict new state vector without heading
			pred_x = this->particles[i].x + velocity*delta_t*cos(this->particles[i].theta);
			pred_y = this->particles[i].y + velocity*delta_t*sin(this->particles[i].theta);
			pred_theta = this->particles[i].theta;
		}
		else
		{
			// predict new state vector with heading influence
			pred_x = this->particles[i].x + velocity/yaw_rate*(sin(this->particles[i].theta + yaw_rate*delta_t) - sin(this->particles[i].theta));
			pred_y = this->particles[i].y + velocity/yaw_rate*(cos(this->particles[i].theta) - cos(this->particles[i].theta + yaw_rate*delta_t));
			pred_theta = this->particles[i].theta + yaw_rate * delta_t;
		}

		// normal distributions for sensor noise
		normal_distribution<double> std_x(pred_x, std_pos[0]);
		normal_distribution<double> std_y(pred_y, std_pos[1]);
		normal_distribution<double> std_theta(pred_theta, std_pos[2]);

		//set particles
		this->particles[i].x = std_x(gen);
		this->particles[i].y = std_y(gen);
		this->particles[i].theta = std_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (unsigned int i = 0; i < observations.size(); i++)
	{
		double dist_min = numeric_limits<double>::max();
		int map_id = -1;
		double obs_x = observations[i].x;
		double obs_y = observations[i].y;

		for (unsigned int j = 0; j < predicted.size(); j++)
		{
			double pred_x = predicted[j].x;
	 		double pred_y = predicted[j].y;

	 		// l2 dist between obs and pred
			double distance = dist(obs_x, obs_y, pred_x, pred_y);

			// find matching landmark
			if (distance < dist_min)
			{
				dist_min = distance;
				map_id = predicted[j].id;
			}
		}

		// set the observation id to found landmark id
		observations[i].id = map_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double gauss_normalizer = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);

	for (unsigned int i = 0; i < this->num_particles; i++)
	{
		double px = this->particles[i].x;
		double py = this->particles[i].y;
		double pt = this->particles[i].theta;

		// Transform observations to the map coordinate system
		vector<LandmarkObs> transformedObs;
		for (unsigned int j = 0; j < observations.size(); j++)
		{
			LandmarkObs transformed_ob;
			transformed_ob.id = observations[j].id;
			transformed_ob.x = px + (cos(pt) * observations[j].x) - (sin(pt) * observations[j].y);
			transformed_ob.y = py + (sin(pt) * observations[j].x) + (cos(pt) * observations[j].y);
			transformedObs.push_back(transformed_ob);
		}

		// Write only the landmarks into a new vector that are in sensor range
		vector<LandmarkObs> filtered_landmarks;
		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++)
		{
			int mark_id = map_landmarks.landmark_list[j].id_i;
			double mark_x = map_landmarks.landmark_list[j].x_f;
			double mark_y = map_landmarks.landmark_list[j].y_f;
			if (dist(px, py, mark_x, mark_y) <= sensor_range)
			{
				filtered_landmarks.push_back(LandmarkObs {mark_id, mark_x, mark_y});
			}
		}

		// Perform dara association
		dataAssociation(filtered_landmarks, transformedObs);

		// Calculate the new weights with the multivariant gaussian method
		this->particles[i].weight = 1.0;
		for (unsigned int j = 0; j < transformedObs.size(); j++)
		{
			double predLandmark_x, predLandmark_y;
			for (unsigned int k = 0; k < filtered_landmarks.size(); k++)
			{
        		if (filtered_landmarks[k].id == transformedObs[j].id)
        		{
          			predLandmark_x = filtered_landmarks[k].x;
          			predLandmark_y = filtered_landmarks[k].y;

          			double multi_gauss = gauss_normalizer * exp( -1.0 * 
          				(pow(predLandmark_x - transformedObs[j].x, 2) / 
          					(2 * pow(std_landmark[0], 2)) + 
          					(pow(predLandmark_y - transformedObs[j].y, 2) / 
          						(2*pow(std_landmark[1], 2))
          					) 
          				) 
          			);

					this->particles[i].weight *= multi_gauss;
        		}
			}			
		}

		//reset our weights 
		this->weights[i] = this->particles[i].weight;
	}
}

void ParticleFilter::resample()
{
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	discrete_distribution<int> discrete_distribution(this->weights.begin(), this->weights.end());

	//reset our particles with a resampled new set
	vector<Particle> resampled;
	for(unsigned int i = 0; i < this->num_particles; i++)
	{
		resampled.push_back(this->particles[discrete_distribution(gen)]);
	}
	this->particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
