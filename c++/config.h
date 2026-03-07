#ifndef CONFIG_H
#define CONFIG_H

/**
 * @file config.h
 * @brief Global configuration parameters for the simulation
 * 
 * This file contains global constants and configuration parameters
 * that control the behavior of the simulation.
 */

// Global simulation dimensions
extern int WIDTH;    /**< Width of the simulation grid */
extern int HEIGHT;   /**< Height of the simulation grid */
extern int SIZE;     /**< Total size of the grid (WIDTH * HEIGHT) */

// Probability parameters
extern double CAPTURE_PROBABILITY;  /**< Probability of successful capture */
extern double HUNTER_PROBABILITY;   /**< Probability for hunter movement behavior */
extern double PREY_PROBABILITY;     /**< Probability for prey movement behavior */

/**
 * @brief Sets the hunter probability parameter
 * 
 * @param newValue New probability value (0.0 to 1.0)
 */
void setHunterProbability(double newValue);

/**
 * @brief Sets the prey probability parameter
 * 
 * @param newValue New probability value (0.0 to 1.0)
 */
void setPreyProbability(double newValue);

#endif // CONFIG_H