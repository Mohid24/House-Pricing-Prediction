#include<iostream>
#include<Eigen/Dense>
#include<fstream>
#include<vector>
#include<iomanip>
using namespace std;
using namespace Eigen;

const int rows = 2000;
const int cols = 6; 

Matrix <double,rows,1> pricevalue;
Matrix <double,rows,cols> matrix;
Matrix <double,cols,rows> transpose;
Matrix <double,cols,cols> Multiply1;
Matrix <double,cols,cols> mulinv;
Matrix <double,cols,rows> Multiply2;
Matrix <double,cols,1> bvariable;

int main()
{
   ifstream fin;
   
   fin.open("House Price Dataset(1).csv");
   
   if(fin)
    {  
      // ─── Loading Screen ───────────────────────────────────────────
      cout << "\n";
      cout << "  ╔══════════════════════════════════════════════════════╗\n";
      cout << "  ║        HOUSE PRICE PREDICTION SYSTEM  v1.0          ║\n";
      cout << "  ║           Linear Regression  |  C++                 ║\n";
      cout << "  ╚══════════════════════════════════════════════════════╝\n\n";
      cout << "  [ 1/3 ]  Loading dataset  ...  ";

      // Matrix Population
      for ( int i = 0 ; i < rows ; i ++ )
        matrix (i,0) = 1;
   
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
      cout << "Done  (" << rows << " records)\n";

      // ─── Training ─────────────────────────────────────────────────
      cout << "  [ 2/3 ]  Training model   ...  ";
      
      transpose = matrix.transpose();
      Multiply1 = transpose * matrix;
      mulinv    = Multiply1.inverse();
      Multiply2 = mulinv * transpose;
      bvariable = Multiply2 * pricevalue;
      
      cout << "Done\n";

      // ─── MSE Calculation ──────────────────────────────────────────
      cout << "  [ 3/3 ]  Evaluating model ...  ";

      double mse = 0.0;
      for ( int i = 0 ; i < rows ; i++ )
      {
          double predicted = bvariable(0,0)
                           + matrix(i,1) * bvariable(1,0)
                           + matrix(i,2) * bvariable(2,0)
                           + matrix(i,3) * bvariable(3,0)
                           + matrix(i,4) * bvariable(4,0)
                           + matrix(i,5) * bvariable(5,0);
          double error = pricevalue(i,0) - predicted;
          mse += error * error;
      }
      mse = mse / rows;
      cout << "Done\n";

      // ─── Model Summary ────────────────────────────────────────────
      cout << "\n";
      cout << "  ┌────────────────────────────────────────────────────┐\n";
      cout << "  │                  Model  Ready                      │\n";
      cout << "  │                                                    │\n";
      cout << fixed << setprecision(2);
      cout << "  │   Mean Squared Error (MSE) :  " << setw(18) << mse << "   │\n";
      cout << "  └────────────────────────────────────────────────────┘\n";

      // ─── User Input ───────────────────────────────────────────────
      double area = 0.0;
      int bedrooms = 0, floors = 0, location = 0, condition = 0;

      cout << "\n";
      cout << "  ╔══════════════════════════════════════════════════════╗\n";
      cout << "  ║              Enter  Property  Details               ║\n";
      cout << "  ╚══════════════════════════════════════════════════════╝\n\n";

      // Area
      cout << "  Area of the House  [ 500 – 9999  sq ft ]  :  ";
      cin >> area;
      while (area < 500 || area > 9999)
      {
        cout << "  ✗  Invalid!  Please enter a value between 500 and 9999  :  ";
        cin >> area;
      }

      // Bedrooms
      cout << "\n  Number of Bedrooms  [ 1 – 5 ]  :  ";
      cin >> bedrooms;
      while (bedrooms < 1 || bedrooms > 5)
      {
        cout << "  ✗  Invalid!  Please enter a value between 1 and 5  :  ";
        cin >> bedrooms;
      }

      // Floors
      cout << "\n  Number of Floors  [ 1 – 3 ]  :  ";
      cin >> floors;
      while (floors < 1 || floors > 3)
      {
        cout << "  ✗  Invalid!  Please enter a value between 1 and 3  :  ";
        cin >> floors;
      }

      // Location
      cout << "\n  Location of the House :\n";
      cout << "    1  →  Downtown\n";
      cout << "    2  →  Urban\n";
      cout << "    3  →  Suburban\n";
      cout << "    4  →  Rural\n";
      cout << "\n  Your Choice  [ 1 – 4 ]  :  ";
      cin >> location;
      while (location < 1 || location > 4)
      {
        cout << "  ✗  Invalid!  Please enter a value between 1 and 4  :  ";
        cin >> location;
      }

      // Condition
      cout << "\n  Condition of the House :\n";
      cout << "    1  →  Excellent\n";
      cout << "    2  →  Good\n";
      cout << "    3  →  Fair\n";
      cout << "    4  →  Poor\n";
      cout << "\n  Your Choice  [ 1 – 4 ]  :  ";
      cin >> condition;
      while (condition < 1 || condition > 4)
      {
        cout << "  ✗  Invalid!  Please enter a value between 1 and 4  :  ";
        cin >> condition;
      }

      // ─── Result ───────────────────────────────────────────────────
      double result = bvariable(0,0) 
                    + (area      * bvariable(1,0))
                    + (bedrooms  * bvariable(2,0))
                    + (floors    * bvariable(3,0))
                    + (location  * bvariable(4,0))
                    + (condition * bvariable(5,0));

      cout << "\n";
      cout << "  ╔══════════════════════════════════════════════════════╗\n";
      cout << "  ║              Prediction  Result                     ║\n";
      cout << "  ║                                                     ║\n";
      cout << "  ║   Estimated House Price  :  $"
           << setw(20) << fixed << setprecision(2) << result << "       ║\n";
      cout << "  ║                                                     ║\n";
      cout << "  ╚══════════════════════════════════════════════════════╝\n\n";

      return 0;
    } 
    else 
    {
      cout << "\n  [ERROR]  Could not open the dataset file.\n";
      cout << "  Make sure  \"House Price Dataset(1).csv\"  is in the same folder.\n\n";
      return 0;
    }
}
