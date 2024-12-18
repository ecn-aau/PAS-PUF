# PAS-PUF

Code to run simulated ML modeling attacks on a generic verifier-prover PUF authentication system that incorporates a Predictive Adversarial System (PAS).

Currently the following PUFs are supported:
 - Arbiter PUF (variable challenge length)
 - XOR PUF (variable chanllenge length and number of parallel Arbiter PUFs)

Currently the following ML algorithms are implemented for PUF modeling and PAS (optimized automatically for the supported PUFs):
 - Logistic Regression (LR)
 - Multi-Layer Perceptron (MLP)

## Dependencies

The code was tested using the following:
 - Python 3.8
 - pypuf 2.2.0
 - numpy 1.23.1
 - tensorflow 2.4.4
 - pandas 1.5.1

## How to use

You can run simulation with one of the provided scripts (use `--help` for a list of arguments):
```
python3 predictive_random_CRP_test.py
python3 predictive_exhaustive_CRP_test.py
python3 predictive_selective_CRP_test.py
```
The keywords *random*, *exhaustive*, and *selective* indicate random, exhaustive and selective (BP, RSP, or BSP) [2] search for CRPs.
In case of selective CRPs, the type is a parameters than can be selected and is RSP with pattern length of 16-bits by default.
Both the verifier with the PAS and the attacker will use independent instances of LR or MLP.

To run a simulation were the verifier run both LR and MLP on the PAS use:
```
python3 predictive_multi_random_CRP_test.py
```
