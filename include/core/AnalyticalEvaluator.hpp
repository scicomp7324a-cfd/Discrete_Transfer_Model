#pragma once

#include <cmath>
#include <numbers>
#include <stdexcept>

using namespace std ;

class AnalyticalEvaluator
{
private:
    // Inputs
    double L_;                 // square plate side length [m]
    double d_;                 // distance between plates [m]
    double A1_;                // plate 1 area [m2]
    double A2_;                // plate 2 area [m2]
    double emissivity1_;       // plate 1 emissivity [-]
    double emissivity2_;       // plate 2 emissivity [-]

    // Constants
    static constexpr double sigma_ = 5.670374419e-8;
    static constexpr double pi_ = 3.14159265358979323846;
    
    // View factors
    double viewFactor12_ = 0.0;     // F12
    double viewFactor21_ = 0.0;     // F21 = A1/A2 * F12
    double viewFactor1Inf_ = 0.0;   // F1_surrounding = 1 - F12
    double viewFactor2Inf_ = 0.0;   // F2_surrounding = 1 - F21

    // Radiosity / irradiation
    double J1_ = 0.0;          // plate 1 radiosity 
    double J2_ = 0.0;          // plate 2 radiosity 
    double G1_ = 0.0;          // plate 1 irradiation 
    double G2_ = 0.0;          // plate 2 irradiation 

    // Total net heat flux leaving each plate, q'' = J - G 
    double q1Total_ = 0.0;
    double q2Total_ = 0.0;

    // Total net heat rate leaving each plate, Q = A q'' 
    double Q1Total_ = 0.0;
    double Q2Total_ = 0.0;

    // Plate-to-plate exchange [W]
    double Q12Gross_ = 0.0;    // gross radiation from 1 to 2
    double Q21Gross_ = 0.0;    // gross radiation from 2 to 1
    double Q12Net_ = 0.0;      // net from 1 to 2 = Q12Gross - Q21Gross

    // Heat loss to surrounding [W]
    double Q1Surrounding_ = 0.0;
    double Q2Surrounding_ = 0.0;

    // Heat loss flux to surrounding [W/m2]
    double q1Surrounding_ = 0.0;
    double q2Surrounding_ = 0.0;

public:
    AnalyticalEvaluator(double L, double d, double A1, double A2,
                        double emissivity1, double emissivity2)
        : L_(L),
          d_(d),
          A1_(A1),
          A2_(A2),
          emissivity1_(emissivity1),
          emissivity2_(emissivity2)
    {
        if (L_ <= 0.0 || d_ <= 0.0 || A1_ <= 0.0 || A2_ <= 0.0){
            throw runtime_error("AnalyticalEvaluator: L, d, A1 and A2 must be positive.");
        }

        if (emissivity1_ <= 0.0 || emissivity1_ > 1.0 || emissivity2_ <= 0.0 || emissivity2_ > 1.0){
            throw runtime_error("AnalyticalEvaluator: emissivity must be in (0, 1].");
        }

        calculateViewFactor() ;
    }

    // ---- getters ----
    double viewFactor12() const { return viewFactor12_; }
    double viewFactor21() const { return viewFactor21_; }
    double viewFactor1Inf() const { return viewFactor1Inf_; }
    double viewFactor2Inf() const { return viewFactor2Inf_; }

    double J1() const { return J1_; }
    double J2() const { return J2_; }
    double G1() const { return G1_; }
    double G2() const { return G2_; }

    double q1Total() const { return q1Total_; }
    double q2Total() const { return q2Total_; }
    double Q1Total() const { return Q1Total_; }
    double Q2Total() const { return Q2Total_; }

    double Q12Gross() const { return Q12Gross_; }
    double Q21Gross() const { return Q21Gross_; }
    double Q12Net() const { return Q12Net_; }

    double Q1Surrounding() const { return Q1Surrounding_; }
    double Q2Surrounding() const { return Q2Surrounding_; }
    double q1Surrounding() const { return q1Surrounding_; }
    double q2Surrounding() const { return q2Surrounding_; }

    // Backward-compatible names used in your older code.
    double viewFactor() const { return viewFactor12_; }
    double q1() const { return q1Total_; }
    double q2() const { return q2Total_; }
    double Q12() const { return Q12Net_; }
    double Q21() const { return -Q12Net_; }

    void calculateViewFactor()
    {
        const double S = L_ / d_;

        const double coeff = 2.0 / (pi_ * S * S);
        const double firstTerm =
            log(sqrt(pow(1.0 + S * S, 2.0) / (1.0 + 2.0 * S * S)));

        const double secondTerm =
            2.0 * S * sqrt(1.0 + S * S) *
            atan(S / sqrt(1.0 + S * S));

        const double thirdTerm = 2.0 * S * atan(S);

        viewFactor12_ = coeff * (firstTerm + secondTerm - thirdTerm);

        // Reciprocity: A1 F12 = A2 F21
        viewFactor21_ = (A1_ / A2_) * viewFactor12_;

        viewFactor1Inf_ = 1.0 - viewFactor12_;
        viewFactor2Inf_ = 1.0 - viewFactor21_;

        // Small numerical safety
        if (viewFactor1Inf_ < 0.0 && viewFactor1Inf_ > -1e-12) viewFactor1Inf_ = 0.0;
        if (viewFactor2Inf_ < 0.0 && viewFactor2Inf_ > -1e-12) viewFactor2Inf_ = 0.0;
    }

    void calculateHeatFluxAndTransfer(double T1, double T2, double Tinf = 0.0)
    {
        //calculateViewFactor();

        const double Eb1 = sigma_ * pow(T1, 4.0);
        const double Eb2 = sigma_ * pow(T2, 4.0);
        const double EbInf = sigma_ * pow(Tinf, 4.0);

        const double r1 = 1.0 - emissivity1_;
        const double r2 = 1.0 - emissivity2_;

        const double C1 = emissivity1_ * Eb1 + r1 * viewFactor1Inf_ * EbInf;
        const double C2 = emissivity2_ * Eb2 + r2 * viewFactor2Inf_ * EbInf;

        const double a = r1 * viewFactor12_;
        const double b = r2 * viewFactor21_;

        const double denom = 1.0 - a * b;

        J1_ = (C1 + a * C2) / denom;
        J2_ = (C2 + b * C1) / denom;

        G1_ = viewFactor12_ * J2_ + viewFactor1Inf_ * EbInf;
        G2_ = viewFactor21_ * J1_ + viewFactor2Inf_ * EbInf;

        // 1. Total net heat flux on each plate
        q1Total_ = J1_ - G1_;
        q2Total_ = J2_ - G2_;

        Q1Total_ = A1_ * q1Total_;
        Q2Total_ = A2_ * q2Total_;

        // 2. Heat loss to surroundings
        Q1Surrounding_ = A1_ * viewFactor1Inf_ * (J1_ - EbInf);
        Q2Surrounding_ = A2_ * viewFactor2Inf_ * (J2_ - EbInf);

        q1Surrounding_ = Q1Surrounding_ / A1_;
        q2Surrounding_ = Q2Surrounding_ / A2_;

        // 3. Heat transfer between both plates
        Q12Gross_ = A1_ * viewFactor12_ * J1_;
        Q21Gross_ = A2_ * viewFactor21_ * J2_;
        Q12Net_ = Q12Gross_ - Q21Gross_;
    }

    double calculatePlateToPlateOnlyNetHeatTransfer(double T1, double T2)
    {
        //calculateViewFactor();

        const double numerator = sigma_ * (pow(T1, 4.0) - pow(T2, 4.0));

        const double denominator =
            ((1.0 - emissivity1_) / (A1_ * emissivity1_)) +
            (1.0 / (A1_ * viewFactor12_)) +
            ((1.0 - emissivity2_) / (A2_ * emissivity2_));

        return numerator / denominator; // W, positive means 1 -> 2
    }
    
};
