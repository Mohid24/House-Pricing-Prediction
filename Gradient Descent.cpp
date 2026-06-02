#include <iostream>
#include <Eigen/Dense>
#include <fstream>

using namespace std;
using namespace Eigen;

// Total records in dataset
const int rows = 2000;

// 1 Bias Column + 5 Features
const int cols = 6;

// Raw dataset storage
double original[rows][cols];

// Actual house prices
double price[rows][1];

// Target variable Y
Matrix<double, rows, 1> pricevalue;

// Feature Matrix X
Matrix<double, rows, cols> matrix;

// Transpose of X
Matrix<double, cols, rows> transpose;

// Regression coefficients
// b0,b1,b2,b3,b4,b5
Matrix<double, cols, 1> bvariable;

int main()
{
    ifstream fin("House Price Dataset(1).csv");

    if (!fin)
    {
        cout << "Dataset could not be opened!" << endl;
        return 0;
    }

    // -----------------------------------
    // Dataset Loading
    // -----------------------------------

    for (int i = 0; i < rows; i++)
    {
        // Bias column = 1
        original[i][0] = 1;
    }

    for (int i = 0; i < rows; i++)
    {
        for (int j = 1; j < cols; j++)
        {
            fin >> original[i][j];
            fin.ignore();
        }

        fin >> price[i][0];
    }

    fin.close();

    // -----------------------------------
    // Move Array Data to Eigen Matrices
    // -----------------------------------

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            matrix(i, j) = original[i][j];
        }
    }

    for (int i = 0; i < rows; i++)
    {
        pricevalue(i, 0) = price[i][0];
    }

    // -----------------------------------
    // Gradient Descent Variables
    // -----------------------------------

    transpose = matrix.transpose();

    // All weights start from zero
    bvariable.setZero();

    // Predicted prices
    Matrix<double, rows, 1> prediction;

    // Prediction errors
    Matrix<double, rows, 1> error;

    // Gradient values
    Matrix<double, cols, 1> gradient;

    // Learning rate
    double learningRate = 0.00000001;

    // Training cycles
    int iterations = 5000;

    // Mean Squared Error
    double mse = 0;

    cout << "Training Model Using Gradient Descent...\n\n";

    // -----------------------------------
    // Gradient Descent Training
    // -----------------------------------

    for (int iter = 0; iter < iterations; iter++)
    {
        // Predict prices
        prediction = matrix * bvariable;

        // Calculate error
        error = prediction - pricevalue;

        // Calculate gradient
        gradient = (transpose * error) / rows;

        // Update coefficients
        bvariable =
            bvariable -
            (learningRate * gradient);

        // Show MSE every 500 iterations
        if (iter % 500 == 0)
        {
            mse =
                error.array().square().sum()
                / rows;

            cout << "Iteration "
                 << iter
                 << " | MSE = "
                 << mse
                 << endl;
        }
    }

    // -----------------------------------
    // Display Learned Model
    // -----------------------------------

    cout << "\nTraining Complete!\n\n";

    cout << "Learned Coefficients:\n";
    cout << "b0 = " << bvariable(0,0) << endl;
    cout << "b1 = " << bvariable(1,0) << endl;
    cout << "b2 = " << bvariable(2,0) << endl;
    cout << "b3 = " << bvariable(3,0) << endl;
    cout << "b4 = " << bvariable(4,0) << endl;
    cout << "b5 = " << bvariable(5,0) << endl;

    // -----------------------------------
    // User Input Section
    // -----------------------------------

    double size;
    int bedrooms;
    int bathrooms;
    int floors;
    int age;

    cout << "\n======================================" << endl;
    cout << " HOUSE PRICE PREDICTION SYSTEM" << endl;
    cout << "======================================\n";

    cout << "Enter House Size : ";
    cin >> size;

    cout << "Enter Bedrooms : ";
    cin >> bedrooms;

    cout << "Enter Bathrooms : ";
    cin >> bathrooms;

    cout << "Enter Floors : ";
    cin >> floors;

    cout << "Enter Age : ";
    cin >> age;

    // -----------------------------------
    // Prediction
    // -----------------------------------

    double result =
        bvariable(0,0)
        + size * bvariable(1,0)
        + bedrooms * bvariable(2,0)
        + bathrooms * bvariable(3,0)
        + floors * bvariable(4,0)
        + age * bvariable(5,0);

    cout << "\nPredicted House Price = "
         << result
         << endl;

    // -----------------------------------
    // Final Training Error
    // -----------------------------------

    prediction = matrix * bvariable;

    error = prediction - pricevalue;

    mse =
        error.array().square().sum()
        / rows;

    cout << "\nFinal Mean Squared Error = "
         << mse
         << endl;

    return 0;
}