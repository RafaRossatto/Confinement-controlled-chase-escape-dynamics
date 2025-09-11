/**
 * @file config.cpp
 * @brief Implementation of global configuration parameters
 */

 #include "config.h"

 // Initialize global configuration variables
 int WIDTH = 0;      /**< Width will be set during initialization */
 int HEIGHT = 0;     /**< Height will be set during initialization */
 int SIZE = 0;       /**< Size will be calculated during initialization */
 
 double CAPTURE_PROBABILITY = 1.00;      /**< Default capture probability */
 double HUNTER_PROBABILITY = 1.00;       /**< Default hunter behavior probability */
 double PREY_PROBABILITY = 1.00;         /**< Default prey behavior probability */
 
 /**
  * @brief Sets the hunter probability parameter
  * 
  * @param newValue New probability value (0.0 to 1.0)
  */
 void setHunterProbability(double newValue)
 {
     if (newValue >= 0.0 && newValue <= 1.0) {
         HUNTER_PROBABILITY = newValue;
     }
     // Optional: Add error handling for invalid values
 }
 
 /**
  * @brief Sets the prey probability parameter
  * 
  * @param newValue New probability value (0.0 to 1.0)
  */
 void setPreyProbability(double newValue)
 {
     if (newValue >= 0.0 && newValue <= 1.0) {
         PREY_PROBABILITY = newValue;
     }
     // Optional: Add error handling for invalid values
 }