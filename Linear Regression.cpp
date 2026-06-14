#include<iostream>
#include<Eigen/Dense>
#include<fstream>
#include<vector>
#include<iomanip>
using namespace std;
using namespace Eigen;

const int rows = 50000;
// 1 (intercept) + 16 numeric features + 2 location dummies + 2 income dummies = 21
const int cols = 21;

MatrixXd pricevalue(rows,1);
MatrixXd matrix(rows,cols);
MatrixXd transpose(cols,rows);
MatrixXd Multiply1(cols,cols);
MatrixXd mulinv(cols,cols);
MatrixXd Multiply2(cols,rows);
MatrixXd bvariable(cols,1);

int main()
{
   ifstream fin;

   fin.open("house_price_50k 1.csv");

   if(fin)
    {
      // ─── Loading Screen ───────────────────────────────────────────
      cout << "\n";
      cout << "  ╔══════════════════════════════════════════════════════╗\n";
      cout << "  ║        HOUSE PRICE PREDICTION SYSTEM  v2.0          ║\n";
      cout << "  ║           Linear Regression  |  C++                 ║\n";
      cout << "  ╚══════════════════════════════════════════════════════╝\n\n";
      cout << "  [ 1/3 ]  Loading dataset  ...  ";

      // Matrix Population
      // Columns 1..16 are read directly from the dataset (numeric features).
      // Columns 17..18 in the CSV (location, income_level) are 1/2/3 codes
      // and are converted into one-hot dummy variables (columns 17..20),
      // with "low" (code 3) as the baseline category for both.
      for ( int i = 0 ; i < rows ; i ++ )
        matrix(i,0) = 1;

      for ( int i = 0 ; i < rows ; i ++ )
      {
        for ( int j = 1 ; j <= 16 ; j ++ )
        {
          fin >> matrix(i,j);
          fin.ignore();
        }

        int location_code = 0, income_code = 0;
        fin >> location_code;
        fin.ignore();
        fin >> income_code;
        fin.ignore();

      // location dummies: premium(1) / mid(2)  -> low(3) is baseline
         if(location_code == 1)
        {
             matrix(i,17) = 1;
             matrix(i,18) = 0;
        } 
        else if(location_code == 2) 
        {
             matrix(i,17) = 0;
             matrix(i,18) = 1;
        }
        else
        {
             matrix(i,17) = 0;
             matrix(i,18) = 0;
        }  
        // income dummies: high(1) / mid(2) -> low(3) is baseline
          if(income_code == 1) 
        {
            matrix(i,19) = 1;
            matrix(i,20) = 0;   
        }
        else if (income_code == 2)
        {
            matrix(i,19) = 0;
            matrix(i,20) = 1;   
        } 
        else
        {
            matrix(i,19) = 0;
            matrix(i,20) = 0;   
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
          double predicted = (matrix.row(i) * bvariable)(0,0);
          double error = pricevalue(i,0) - predicted;
          mse += error * error;
      }
      mse = mse / rows;
      cout << "Done\n";

      // ─── Model Summary ────────────────────────────────────────────
      cout << "\n";
      cout << "  ┌────────────────────────────────────────────────────┐\n";
      cout << "  │                  Model  Ready                       │\n";
      cout << "  │                                                      │\n";
      cout << fixed << setprecision(2);
      cout << "  │   Mean Squared Error (MSE) :  " << setw(18) << mse << "   │\n";
      cout << "  └────────────────────────────────────────────────────┘\n";

      // ─── User Input ───────────────────────────────────────────────
      double area = 0.0, crimerate = 0.0, population_density = 0.0;
      int bedrooms = 0, bathroom = 0, age = 0, distance = 0, garage = 0;
      int parking = 0, floors = 0, location = 0, income_level = 0;
      int garden = 0, security = 0, school_nearby = 0;
      int hospital_nearby = 0, shopping_mall_nearby = 0, transport = 0;

      cout << "\n";
      cout << "  ╔══════════════════════════════════════════════════════╗\n";
      cout << "  ║              Enter  Property  Details               ║\n";
      cout << "  ╚══════════════════════════════════════════════════════╝\n\n";

      // Area
      cout << "  Area of the House  [ 500 - 9999  sq ft ]  :  ";
      cin >> area;
      while (area < 500 || area > 9999)
      {
        cout << "  X  Invalid! Please enter a value between 500 and 9999  :  ";
        cin >> area;
      }

      // Bedrooms
      cout << "\n  Number of Bedrooms  [ 1 - 10 ]  :  ";
      cin >> bedrooms;
      while (bedrooms < 1 || bedrooms > 10)
      {
        cout << "  X  Invalid! Please enter a value between 1 and 10  :  ";
        cin >> bedrooms;
      }

      // Bathroom
      cout << "\n  Number of Bathrooms  [ 1 - 10 ]  :  ";
      cin >> bathroom;
      while (bathroom < 1 || bathroom > 10)
      {
        cout << "  X  Invalid! Please enter a value between 1 and 10  :  ";
        cin >> bathroom;
      }

      // Floors
      cout << "\n  Number of Floors  [ 1 - 3 ]  :  ";
      cin >> floors;
      while (floors < 1 || floors > 3)
      {
        cout << "  X  Invalid! Please enter a value between 1 and 3  :  ";
        cin >> floors;
      }

      // Age
      cout << "\n  Age of the house  [ 1 - 50 ]  :  ";
      cin >> age;
      while (age < 1 || age > 50)
      {
        cout << "  X  Invalid! Please enter a value between 1 and 20  :  ";
        cin >> age;
      }

      // Distance
      cout << "\n  Distance  [ 1 - 50 ]  :  ";
      cin >> distance;
      while (distance < 1 || distance > 50)
      {
        cout << "  X  Invalid! Please enter a value between 1 and 50  :  ";
        cin >> distance;
      }

      // Garage
      cout << "\n  Have a garage? Yes/No [ 1/0 ]  :  ";
      cin >> garage;
      while (garage < 0 || garage > 1)
      {
        cout << "  X  Invalid! Please enter 1 or 0  :  ";
        cin >> garage;
      }

      // Parking
      cout << "\n  Have parking? Yes/No [ 1/0 ]  :  ";
      cin >> parking;
      while (parking < 0 || parking > 1)
      {
        cout << "  X  Invalid! Please enter 1 or 0  :  ";
        cin >> parking;
      }

      // Garden
      cout << "\n  Have a garden? Yes/No [ 1/0 ]  :  ";
      cin >> garden;
      while (garden < 0 || garden > 1)
      {
        cout << "  X  Invalid! Please enter 1 or 0  :  ";
        cin >> garden;
      }

      // Security
      cout << "\n  Tight security? Yes/No [ 1/0 ]  :  ";
      cin >> security;
      while (security < 0 || security > 1)
      {
        cout << "  X  Invalid! Please enter 1 or 0  :  ";
        cin >> security;
      }

      // School nearby
      cout << "\n  School nearby? Yes/No [ 1/0 ]  :  ";
      cin >> school_nearby;
      while (school_nearby < 0 || school_nearby > 1)
      {
        cout << "  X  Invalid! Please enter 1 or 0  :  ";
        cin >> school_nearby;
      }

      // Hospital nearby
      cout << "\n  Hospital nearby? Yes/No [ 1/0 ]  :  ";
      cin >> hospital_nearby;
      while (hospital_nearby < 0 || hospital_nearby > 1)
      {
        cout << "  X  Invalid! Please enter 1 or 0  :  ";
        cin >> hospital_nearby;
      }

      // Shopping mall nearby
      cout << "\n  Shopping mall nearby? Yes/No [ 1/0 ]  :  ";
      cin >> shopping_mall_nearby;
      while (shopping_mall_nearby < 0 || shopping_mall_nearby > 1)
      {
        cout << "  X  Invalid! Please enter 1 or 0  :  ";
        cin >> shopping_mall_nearby;
      }

      // Public transport
      cout << "\n  Public transport nearby? Yes/No [ 1/0 ]  :  ";
      cin >> transport;
      while (transport < 0 || transport > 1)
      {
        cout << "  X  Invalid! Please enter 1 or 0  :  ";
        cin >> transport;
      }

      // Crime rate
      cout << "\n  Crime Rate  [ 0 - 10 ]  :  ";
      cin >> crimerate;
      while (crimerate < 0 || crimerate > 10)
      {
        cout << "  X  Invalid! Please enter a value between 0 and 10  :  ";
        cin >> crimerate;
      }

      // Population density
      cout << "\n  Population density  [ 500 - 9999 ]  :  ";
      cin >> population_density;
      while (population_density < 500 || population_density > 9999)
      {
        cout << "  X  Invalid! Please enter a value between 500 and 9999  :  ";
        cin >> population_density;
      }

      // Location
      cout << "\n  Location of the House :\n";
      cout << "    1  ->  Premium\n";
      cout << "    2  ->  Mid\n";
      cout << "    3  ->  Low\n";
      cout << "\n  Your Choice  [ 1 - 3 ]  :  ";
      cin >> location;
      while (location < 1 || location > 3)
      {
        cout << "  X  Invalid! Please enter a value between 1 and 3  :  ";
        cin >> location;
      }

      // Income level
      cout << "\n  Income level of the area :\n";
      cout << "    1  ->  High\n";
      cout << "    2  ->  Mid\n";
      cout << "    3  ->  Low\n";
      cout << "\n  Your Choice  [ 1 - 3 ]  :  ";
      cin >> income_level;
      while (income_level < 1 || income_level > 3)
      {
        cout << "  X  Invalid! Please enter a value between 1 and 3  :  ";
        cin >> income_level;
      }

      // Convert location/income choices to the same dummy encoding used in training
      int loc_premium = 0;
      int loc_mid     = 0;
      int inc_high    = 0;
      int inc_mid     = 0;
     if ( location == 1 )
     {
          loc_premium = 1;
          loc_mid     = 0;
     }
     else if ( location == 2 ) 
     {
          loc_premium = 0;
          loc_mid     = 1;
     }
     else
     {
          loc_premium = 0;
          loc_mid     = 0;
     }
     if ( income_level == 1 )
     {
          inc_high = 1;
          inc_mid  = 0;
     }
     else if ( income_level == 2 ) 
     {
          inc_high = 0;
          inc_mid  = 1;
     }
     else
     {
          inc_high = 0;
          inc_mid  = 0;
     }
      // ─── Result ───────────────────────────────────────────────────
      double result = bvariable(0,0)
                    + (area                  * bvariable(1,0))
                    + (bedrooms              * bvariable(2,0))
                    + (bathroom              * bvariable(3,0))
                    + (floors                * bvariable(4,0))
                    + (age                   * bvariable(5,0))
                    + (distance              * bvariable(6,0))
                    + (garage                * bvariable(7,0))
                    + (parking               * bvariable(8,0))
                    + (garden                * bvariable(9,0))
                    + (security              * bvariable(10,0))
                    + (school_nearby         * bvariable(11,0))
                    + (hospital_nearby       * bvariable(12,0))
                    + (shopping_mall_nearby  * bvariable(13,0))
                    + (transport             * bvariable(14,0))
                    + (crimerate             * bvariable(15,0))
                    + (population_density    * bvariable(16,0))
                    + (loc_premium           * bvariable(17,0))
                    + (loc_mid               * bvariable(18,0))
                    + (inc_high              * bvariable(19,0))
                    + (inc_mid               * bvariable(20,0));

      cout << "\n";
      cout << "  ╔══════════════════════════════════════════════════════╗\n";
      cout << "  ║              Prediction  Result                     ║\n";
      cout << "  ║                                                      ║\n";
      cout << "  ║   Estimated House Price  :  $"
           << setw(18) << fixed << setprecision(2) << result << "       ║\n";
      cout << "  ║                                                      ║\n";
      cout << "  ╚══════════════════════════════════════════════════════╝\n\n";

      return 0;
    }
    else
    {
      cout << "\n  [ERROR]  Could not open the dataset file.\n";
      cout << "  Make sure  \"house_price_50k_1.csv\"  is in the same folder.\n\n";
      return 1;
    }
}
