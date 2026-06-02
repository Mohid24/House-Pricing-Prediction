#include<iostream>
#include<Eigen/Dense>
#include<fstream>
#include<vector>
using namespace std;
using namespace Eigen;

const int rows = 2000;
 const int cols = 6; 
//  double original [rows] [cols];
//  double price [rows] [1]; 
 
Matrix <double,rows,1> pricevalue; // Y  
 // double price [rows] [1]; 2000 X 1
 
 Matrix <double,rows,cols> matrix;  // X 
 //double original [rows] [cols]; 2000 X 6 
 
 Matrix <double,cols,rows> transpose; // X(t)  
 //double transpose [cols] [rows]; 2000 X 6
 
 Matrix <double,cols,cols> Multiply1; // X(t) * X 
 //double multiply1 [cols] [cols];  6 X 6
 
 Matrix <double,cols,cols> mulinv;   // (X(t) * X )(-1) 
 //double inverse [cols] [cols];  6 X 6
 
 Matrix <double,cols,rows> Multiply2; // ([(X(t) * X )(-1)]* X(t))   
 //double multiply2 [cols] [cols]; 6 X 2000
 
 Matrix <double,cols,1> bvariable; // b^  
 //double multiply4 [cols] [1];  6 X 1
int main()
{
   ifstream fin;
   
   // Firstly used 2D Arrays but the inversion of Giant Matrix was difficult so we Use
   // Eigen Library for Matrix Manipulation
   
   fin.open("House Price Dataset(1).csv");
   
   if(fin)
    {  
        // Matrix Population
      for ( int i = 0 ; i < rows ; i ++ )
     
      {  
          matrix (i,0) = 1;
      }
   
      for ( int i = 0 ; i < rows ; i ++ )
   
      {  
   
        for ( int j = 1 ; j < cols ; j ++ )
   
        {
        fin >> matrix (i,j);
        fin.ignore(); 
        }
   
       fin >> pricevalue(i,0);
    
    } 
       fin.close();
     
      // Calculation of Slopes and Interception Variables
     
      // Y = b0 +  X1 * b1  +  X2 * b2  +  X3 * b3  +  X4 * b4 + X5 * b5 
    
      transpose = matrix.transpose(); // Conversion of 2000 X 6 ===> 6 X 2000    
     
      Multiply1 = transpose * matrix; // 6 X 2000 * 2000 X 6 ===> 6 X 6
     
      mulinv = Multiply1.inverse();  // 6 X 6 still
     
      Multiply2 = mulinv * transpose; // 6 X 6 * 6 X 2000 ===> 6 X 2000
     
      bvariable = Multiply2 * pricevalue; // 6 X 2000 * 2000 X 1 ===> 6 X 1 
     
      // b0 b1 b2 b3 b4 b5 
        
     // ======== Mean Squared Error (MSE) Calculation ========
     // MSE = (1/n) * sum( (actual - predicted)^2 )
     // We compute predicted price for each row using bvariable,
     // then compare it with the actual price to measure model accuracy.

      double mse = 0.0; // Accumulator for squared errors

      for ( int i = 0 ; i < rows ; i++ )
      {
          // Compute predicted price for row i using the regression coefficients
          double predicted = bvariable(0,0)                          // b0 (intercept)
                           + matrix(i,1) * bvariable(1,0)        // b1 * Area
                           + matrix(i,2) * bvariable(2,0)        // b2 * Bedrooms
                           + matrix(i,3) * bvariable(3,0)        // b3 * Floors
                           + matrix(i,4) * bvariable(4,0)        // b4 * Location
                           + matrix(i,5) * bvariable(5,0);       // b5 * Condition

          double error = pricevalue(i,0) - predicted; // Actual minus Predicted
          mse += error * error;                   // Accumulate squared error
      }

      mse = mse / rows; // Divide by number of rows to get the mean
      // ======================================================
        
      // Data Input for Parameters
      double area = 0.0 ;
      int  bedrooms = 0 , floors = 0 , location = 0 , condition = 0; 
      
      cout << "======== < Welcome to our C++ Based House Prediction System > ========" << endl;
      
      cout << "Mean Squared Error (MSE) of the Model : " << mse << endl;
      
      cout << "Please enter the Area of the House in Square Feet ( 500 - 9999 ) : ";
      cin >> area;
      
      while (area < 500 || area > 9999 )
      
      {
      cout << "Please Enter the valid range area in ( 500 - 9999 ) : ";
      cin >> area;
      }
      
      cout << "Please Input the Bedrooms in the House ( 1 - 5 ) : ";
      cin >> bedrooms;
      
      while (bedrooms < 1 || bedrooms > 5 )
      
      {
      cout << "Please Enter the valid range of bedrooms in ( 1 - 5 ) : ";
      cin >> bedrooms;
      }
      
      cout << "Please Input the Floors in the House ( 1 - 3 ) : ";
      cin >> floors;
      
      while (floors < 1 || floors > 3 )
      
      {
      cout << "Please Enter the valid range of Floors in ( 1 - 3 ) : ";
      cin >> floors;
      }
      
      cout << "Please Input the Location of the House accordind to following Choices : " << endl;
      cout << "Downtown ==> 1" << endl;
      cout << "Urban ==> 2" << endl;
      cout << "Suburban ==> 3" << endl;
      cout << "Rural ==> 4" << endl;
      cout << "Enter the Choice in b/w (1 - 4) : ";
      cin >> location;
      
      while (location < 1 || location > 4 )
      
      {
      cout << "Please Enter the valid range of Choices in ( 1 - 4 ) : ";
      cin >> location;
      }
      
      cout << "Please Input the Condition of the House accordind to following Choices : " << endl;
      cout << "Excellent ==> 1" << endl;
      cout << "Good ==> 2" << endl;
      cout << "Fair ==> 3" << endl;
      cout << "Poor ==> 4" << endl;
      cout << "Enter the Choice in b/w (1 - 4) : ";
      cin >> condition;
      
      while (condition < 1 || condition > 4 )
      {
      cout << "Please Enter the valid range of Condition in ( 1 - 4 ) : ";
      cin >> condition;
      }
      
      double result = 0.0;
      
      result = bvariable(0,0) + (area * bvariable(1,0)) + ( bedrooms * bvariable(2,0)) + ( floors * bvariable(3,0)) + (location * bvariable(4,0)) + ( condition * bvariable(5,0) ) ; 
      
      cout << "Your Predicted Price Value is : " << result << endl;
      
      return 0 ;
    } 
    else 
    
    {
    
        cout << "Somehow the file is not openning due to me error." << endl;
        return 0;
    
    }
}