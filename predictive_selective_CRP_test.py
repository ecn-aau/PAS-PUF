# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:21:56 2023

@author: Mieszko Ferens

Script to run an experiment for modelling an XOR Arbiter PUF that uses
predictive selective CRP selection during authentication with the server.

Selective CRPs are taken from:
    M. Ferens, E. Dushku and S. Kosta, "Securing PUFs Against ML Modeling
    Attacks via an Efficient Challenge-Response Approach," IEEE INFOCOM 2023 -
    IEEE Conference on Computer Communications Workshops (INFOCOM WKSHPS),
    Hoboken, NJ, USA, 2023, pp. 1-6,
    doi: 10.1109/INFOCOMWKSHPS57453.2023.10226062.
"""

import argparse
import pandas as pd
from pathlib import Path

import numpy as np
from pypuf.simulation import XORArbiterPUF
from pypuf.io import random_inputs
import pypuf.attack

class ChallengeResponseSet():
    def __init__(self, n, challenges, responses):
        self.challenge_length = n
        self.challenges = challenges
        self.responses = np.expand_dims(
            np.expand_dims(responses,axis=1),axis=1)

def create_binary_code_challenges(n, N):
    
    n_bits = 16
    lsb = np.arange(2**n_bits, dtype=np.uint8).reshape(-1,1)
    msb = lsb.copy()
    msb.sort(axis=0)

    lsb = np.unpackbits(lsb, axis=1)[:,-8:].copy()
    msb = np.unpackbits(msb, axis=1)[:,-8:].copy()

    challenges = 2*np.concatenate((msb,lsb), axis=1, dtype=np.int8) - 1
    for i in range(int(np.sqrt(n/n_bits))):
        challenges = np.insert(
            challenges, range(1,((2**i)*n_bits)+1), -1, axis=1)
    
    shift = challenges.copy()
    for i in range(1, int(n/n_bits)):
        challenges = np.append(challenges, np.roll(shift, i, axis=1), axis=0)
    
    _ , idx = np.unique(challenges, return_index=True, axis=0)
    challenges = challenges[np.sort(idx)]
    
    assert N <= len(challenges), (
        "Not enough CRPs have been generated. The limit is (2^18 - 3) CRPs.")
    challenges = challenges[:N]
    
    return challenges

def create_shifted_pattern_challenges(n, N, n_patterns, pattern_len, seed=0):
    
    patterns = random_inputs(pattern_len, n_patterns, seed=seed)

    challenges = -np.ones(((n-pattern_len+1)*n_patterns,n), dtype=np.int8)
    for i in range(n_patterns):
        for j in range(n-pattern_len+1):
            challenges[i*(n-pattern_len+1)+j, j:j+pattern_len] = patterns[i]
    
    _ , idx = np.unique(challenges, return_index=True, axis=0)
    challenges = challenges[np.sort(idx)]
    
    assert N <= len(challenges), (
        "Not enough unique CRPs exist due to duplicates. " +
        "Tip: You might need to increase the number of patterns")
    challenges = challenges[:N]
    
    return challenges

def create_binary_pattern_challenges(n, N, pattern_len):
    
    assert ((2**(pattern_len))*(n-pattern_len+1) > N), (
        "Pattern length is too low for the number of patterns")
    
    max_bits = int(np.ceil(np.log2(N)))
    bits = min(max_bits, pattern_len)
    
    lsb = np.arange(2**bits, dtype=np.uint8).reshape(-1,1)
    
    extra = 0
    if(bits % 8):
        extra = 1
    
    msb = []
    for i in range(1, int(bits/8) + extra):
        msb.append(lsb.copy())
        for j in range(2**8):
            msb[i-1][(2**(8*(i+1)))*j:(2**(8*(i+1)))*(j+1)].sort(axis=0)

    lsb = np.unpackbits(lsb, axis=1)[:,-8:].copy()
    for i in range(len(msb)):
        msb[i] = np.unpackbits(msb[i], axis=1)[:,-8:].copy()

    msb.insert(0, lsb)
    patterns = 2*np.concatenate(msb[::-1], axis=1, dtype=np.int8) - 1
    if(bits % 8):
        patterns = np.delete(
            patterns[:N], slice(8 - (bits % 8)), axis=1)
    else:
        patterns = patterns[:N]
    
    if(bits < pattern_len):
        patterns = np.concatenate(
            (-np.ones((N, pattern_len - bits), dtype=np.int8), patterns),
            axis=1, dtype=np.int8)
    
    challenges = -np.ones(((n-pattern_len+1)*len(patterns), n), dtype=np.int8)
    for i in range(len(patterns)):
        for j in range(n-pattern_len+1):
            challenges[i*(n-pattern_len+1)+j, j:j+pattern_len] = patterns[i]
    
    _ , idx = np.unique(challenges, return_index=True, axis=0)
    challenges = challenges[np.sort(idx)]
    
    assert N <= len(challenges), (
        "Not enough unique CRPs exist due to duplicates. " +
        "Tip: You might need to increase the number of patterns")
    challenges = challenges[:N]
    
    return challenges

def main():
    
    # Set-up logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="./Results/")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--n-bits", type=int, default=64,
                        help="Challenge length in bits.")
    parser.add_argument("--k", type=int, default=1,
                        help="The number of parallel APUF in the XOR PUF.")
    parser.add_argument("--train-data", type=int, default=10000,
                        help="Number of training data samples for the model.")
    parser.add_argument("--auth-CRPs", type=int, default=200,
                        help="Number of CRPs used per authentication query.")
    parser.add_argument("--searches", type=int, default=10,
                        help="Number of sets of CRP for each authentication " +
                        "query to check for modelling accuracy.")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Authentication and viability threshold for CRPs.")
    parser.add_argument("--server-predictor", type=str, default="LR",
                        help="The ML algorithm that the authentication " +
                        "server uses to select CRPs.")
    parser.add_argument("--attacker-predictor", type=str, default="LR",
                        help="The ML algorithm that the attacker uses to " +
                        " attempt to model the PUF.")
    parser.add_argument("--min-auths", type=int, default=5,
                        help="Minimum number of successful authentications" +
                        "for a prover to be considered authentic.")
    parser.add_argument("--selective_CRPs", type=str, default="RSP",
                        help="The type of selective CRPs to used (BP, RSP, " +
                        "or BSP)")
    args = parser.parse_args()
    
    # Generate the PUF
    puf = XORArbiterPUF(args.n_bits, args.k, args.seed)
    
    # Check if training CRPs are divisible by the authentication query size
    if(args.train_data % args.auth_CRPs != 0):
        print("Warning: The number of training CRPs will not be exactly " +
              str(args.train_data) + "!")
    
    # Create and store the full set of CRPs used for authentication
    # Note: pattern_len values taken from reference for Selective CRPs
    if(args.selective_CRPs == "BP"): # Use BP as selective CRPs
        challenge_set = create_binary_code_challenges(
            n=args.n_bits, N=args.train_data)
    elif(args.selective_CRPs == "RSP"): # Use RSP as selective CRPs
        challenge_set = create_shifted_pattern_challenges(
            n=args.n_bits, N=args.train_data, n_patterns=args.train_data,
            pattern_len=16, seed=args.seed)
    elif(args.selective_CRPs == "BSP"): # Use BSP as selective CRPs
        challenge_set = create_binary_pattern_challenges(
            n=args.n_bits, N=args.train_data, pattern_len=25)
    else:
        raise NotImplementedError("Only BP, RSP and BSP selective CRPs are" +
                                  " supported.")
        
    response_set = puf.eval_block(challenge_set)
    challenges = np.array([], dtype=np.int8)
    responses = np.array([], dtype=np.float64)
    count = 0 # Current CRP to check for viability
    auth_count = 0 # Current number of consecutive successful authentications of the attacker
    
    # Start appending challenges that are used for authentication
    challenges = np.append(challenges, challenge_set[:args.auth_CRPs]).reshape(
        -1, args.n_bits)
    responses = np.append(responses, response_set[:args.auth_CRPs])
    count += args.auth_CRPs
    
    # Use a CRP predictor on the authentication server
    if(args.server_predictor == "LR"): # Use LR as a predictor
        model_server = pypuf.attack.LRAttack2021(
            ChallengeResponseSet(args.n_bits, challenges, responses),
            seed=args.seed, k=args.k, epochs=100, lr=.001, bs=1000,
            stop_validation_accuracy=.97)
    elif(args.server_predictor == "MLP"): # Use MLP as a predictor
        if(args.k <= 4): # If the XOR PUF is small don't reduce the NN too much
            network = [8, 16, 8] 
        else: # As defined in the literature: [2^(k-1), 2^k, 2^(k-1)]
            network = [2**(args.k-1), 2**args.k, 2**(args.k-1)]
        model_server = pypuf.attack.MLPAttack2021(
            ChallengeResponseSet(args.n_bits, challenges, responses),
            seed=args.seed, net=network, epochs=30, lr=.001, bs=1000,
            early_stop=.08)
    else:
        raise NotImplementedError("Only LR and MLP are supported.")
    
    # Use a ML algorithm to model PUF on the attacker
    if(args.attacker_predictor == "LR"): # Use LR as a predictor
        model_attacker = pypuf.attack.LRAttack2021(
            ChallengeResponseSet(args.n_bits, challenges, responses),
            seed=args.seed+1, k=args.k, epochs=100, lr=.001, bs=1000,
            stop_validation_accuracy=.97)
    elif(args.attacker_predictor == "MLP"): # Use MLP as a predictor
        if(args.k <= 4): # If the XOR PUF is small don't reduce the NN too much
            network = [8, 16, 8] 
        else: # As defined in the literature: [2^(k-1), 2^k, 2^(k-1)]
            network = [2**(args.k-1), 2**args.k, 2**(args.k-1)]
        model_attacker = pypuf.attack.MLPAttack2021(
            ChallengeResponseSet(args.n_bits, challenges, responses),
            seed=args.seed+1, net=network, epochs=30, lr=.001, bs=1000,
            early_stop=.08)
    else:
        raise NotImplementedError("Only LR and MLP are supported.")
    
    model_server.fit()
    model_attacker.fit()
    
    print("Authenticating " + str(round(args.train_data/args.auth_CRPs)) +
          " times with " + str(args.auth_CRPs) + " CRPs...")
    
    # Prepare log file
    filepath = Path(
        args.outdir + "out_predictive_selective_" + str(args.selective_CRPs) +
        "_CRPs_" + str(args.k) + "XOR.csv")
    if(not filepath.is_file()):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = pd.DataFrame({"seed": [], "n_bits": [], "k": [],
                             "train_data": [], "auth_CRPs": [], "searches": [],
                             "threshold": [], "server_predictor": [],
                             "attacker_predictor": [], "min_auths": [],
                             "selective_CRPs": [], "viable_CRPs": [],
                             "server_accuracy": [], "attacker_accuracy": []})
        data.to_csv(filepath, header=True, index=False, mode='a')
    
    # Authentication queries
    for i in range(1, round(args.train_data/args.auth_CRPs)):
        
        print(" - Auth " + str(i))
        
        for crp in range(args.auth_CRPs):
            
            # print(" -- CRP " + str(crp))
            
            # Search for unpredictable (viable) challenge
            for j in range(args.searches):
                
                # print(" --- Search " + str(j))
                
                # End if selective challenge set run out
                assert count < len(challenge_set), (
                    "Selective CRP set is exhausted!")
                
                # Select new challenge
                challenge = np.expand_dims(challenge_set[count], axis=0)
                response = 0.5 - 0.5*response_set[count]
                count += 1
                
                # Test prediction of model
                pred_y = model_server._model.eval(challenge)
                pred_y = pred_y.reshape(len(pred_y), 1)
                
                # If CRP is unpredictable (viable), else keep searching
                if((((pred_y < 0.5) + response) - 1) == 0):
                    break
            
            # Add next challenge
            challenges = np.append(challenges, challenge).reshape(
                -1, args.n_bits)
            responses = np.append(responses, puf.eval(challenge))
        
        # Test if server predicts false authentication
        test_x = challenges[i*args.auth_CRPs:(i+1)*args.auth_CRPs]
        test_y = np.expand_dims(
            0.5 - 0.5*responses[i*args.auth_CRPs:(i+1)*args.auth_CRPs], axis=1)
        pred_y = model_server._model.eval(test_x)
        pred_y = pred_y.reshape(len(pred_y), 1)
        server_acc = np.count_nonzero(((pred_y<0.5) + test_y)-1)/len(test_y)
        
        # Test if attacker can falsely authenticate with the server
        test_x = challenges[i*args.auth_CRPs:(i+1)*args.auth_CRPs]
        test_y = np.expand_dims(
            0.5 - 0.5*responses[i*args.auth_CRPs:(i+1)*args.auth_CRPs], axis=1)
        pred_y = model_attacker._model.eval(test_x)
        pred_y = pred_y.reshape(len(pred_y), 1)
        attacker_acc = np.count_nonzero(((pred_y<0.5) + test_y)-1)/len(test_y)
        
        # Log data into csv format
        data = pd.DataFrame({"seed": [args.seed],
                             "n_bits": [args.n_bits],
                             "k": [args.k],
                             "train_data": [args.train_data],
                             "auth_CRPs": [args.auth_CRPs],
                             "searches": [args.searches],
                             "threshold": [args.threshold],
                             "server_predictor": [args.server_predictor],
                             "attacker_predictor": [args.attacker_predictor],
                             "min_auths": [args.min_auths],
                             "selective_CRPs": [args.selective_CRPs],
                             "viable_CRPs": [len(challenges)],
                             "server_accuracy": [server_acc],
                             "attacker_accuracy": [attacker_acc]})
        data.to_csv(filepath, header=False, index=False, mode='a')
        
        # Train models
        model_server.crps = ChallengeResponseSet(
            args.n_bits, challenges, responses)
        model_server.fit()
        model_attacker.crps = ChallengeResponseSet(
            args.n_bits, challenges, responses)
        if(attacker_acc >= args.threshold):
            print("Attacker prediction accuracy over threshold!")
            auth_count += 1
            if(auth_count >= args.min_auths):
                print("Attacker impersonation successful!")
                break
        else:
            auth_count = 0
            model_attacker.fit()
    

if(__name__ == "__main__"):
    main()

