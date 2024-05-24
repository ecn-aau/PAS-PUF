# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:02:52 2024

@author: Mieszko Ferens
"""

import argparse
import pandas as pd
from pathlib import Path

import numpy as np
from pypuf.simulation import XORArbiterPUF
import pypuf.attack

class ChallengeResponseSet():
    def __init__(self, n, challenges, responses):
        self.challenge_length = n
        self.challenges = challenges
        self.responses = np.expand_dims(
            np.expand_dims(responses,axis=1),axis=1)

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
    parser.add_argument("--train-data", type=int, default=100000,
                        help="Number of training data samples for the model.")
    parser.add_argument("--auth-CRPs", type=int, default=200,
                        help="Number of CRPs used per authentication query.")
    parser.add_argument("--searches", type=int, default=100,
                        help="Number of sets of CRP for each authentication " +
                        "query to check for modelling accuracy.")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Authentication and viability threshold for CRPs.")
    parser.add_argument("--attacker-predictor", type=str, default="LR",
                        help="The ML algorithm that the attacker uses to " +
                        " attempt to model the PUF.")
    parser.add_argument("--min-auths", type=int, default=1,
                        help="Minimum number of successful authentications" +
                        "for a prover to be considered authentic.")
    args = parser.parse_args()
    
    # Generate the PUF
    puf = XORArbiterPUF(args.n_bits, args.k, args.seed)
    
    # Check if training CRPs are divisible by the authentication query size
    if(args.train_data % args.auth_CRPs != 0):
        print("Warning: The number of training CRPs will not be exactly " +
              str(args.train_data) + "!")
    
    # Create and store the full set of CRPs used for authentication
    challenges = np.array([], dtype=np.int8)
    responses = np.array([], dtype=np.float64)
    auth_count = 0 # Current number of consecutive successful authentications of the attacker
    
    # Create first query
    challenges = np.append(
        challenges,
        (np.random.randint(2, size=args.n_bits*args.auth_CRPs)*2 - 1)).reshape(
            -1, args.n_bits)
    responses = np.append(responses, puf.eval_block(challenges))
    
    # Use a CRP predictor on the authentication server
    # Use LR as a predictor
    model_LR_server = pypuf.attack.LRAttack2021(
        ChallengeResponseSet(args.n_bits, challenges, responses),
        seed=args.seed, k=args.k, epochs=100, lr=.001, bs=1000,
        stop_validation_accuracy=.97)
    # Use MLP as a predictor
    if(args.k <= 4): # If the XOR PUF is small don't reduce the NN too much
        network = [8, 16, 8] 
    else: # As defined in the literature: [2^(k-1), 2^k, 2^(k-1)]
        network = [2**(args.k-1), 2**args.k, 2**(args.k-1)]
    model_MLP_server = pypuf.attack.MLPAttack2021(
        ChallengeResponseSet(args.n_bits, challenges, responses),
        seed=args.seed, net=network, epochs=30, lr=.001, bs=1000,
        early_stop=.08)
    
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
    
    model_LR_server.fit()
    model_MLP_server.fit()
    model_attacker.fit()
    
    print("Authenticating " + str(round(args.train_data/args.auth_CRPs)) +
          " times with " + str(args.auth_CRPs) + " CRPs...")
    
    # Prepare log file
    filepath = Path(
        args.outdir + "out_predictive_random_CRPs_" + str(args.k) +
        "XOR.csv")
    if(not filepath.is_file()):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = pd.DataFrame({"seed": [], "n_bits": [], "k": [],
                             "train_data": [], "auth_CRPs": [], "searches": [],
                             "threshold": [], "server_predictor": [],
                             "attacker_predictor": [], "min_auths": [],
                             "viable_CRPs": [], "server_accuracy": [],
                             "attacker_accuracy": []})
        data.to_csv(filepath, header=True, index=False, mode='a')
    
    # Authentication queries
    for i in range(1, round(args.train_data/args.auth_CRPs)):
        
        print(" - Auth " + str(i))
        
        for crp in range(args.auth_CRPs):
            
            # print(" -- CRP " + str(crp))
            
            # Search for unpredictable (viable) challenge
            for j in range(args.searches):
                
                # print(" --- Search " + str(j))
                
                # Select new challenge
                challenge = np.expand_dims(
                    np.random.randint(2, size=args.n_bits)*2 - 1, axis=0)
                response = np.expand_dims(0.5 - 0.5*puf.eval(challenge), axis=1)
                
                # Test prediction of LR model
                pred_LR_y = model_LR_server._model.eval(challenge)
                pred_LR_y = pred_LR_y.reshape(len(pred_LR_y), 1)
                
                # Test prediction of MLP model
                pred_MLP_y = model_MLP_server._model.eval(challenge)
                pred_MLP_y = pred_MLP_y.reshape(len(pred_MLP_y), 1)
                
                # If CRP is unpredictable (viable), else keep searching
                if((((pred_LR_y < 0.5) + response) - 1) == 0 and
                   (((pred_MLP_y < 0.5) + response) - 1) == 0):
                    break
            
            # Add next challenge
            challenges = np.append(challenges, challenge).reshape(
                -1, args.n_bits)
            responses = np.append(responses, puf.eval(challenge))
        
        # Test if server predicts false authentication
        test_x = challenges[i*args.auth_CRPs:(i+1)*args.auth_CRPs]
        test_y = np.expand_dims(
            0.5 - 0.5*responses[i*args.auth_CRPs:(i+1)*args.auth_CRPs], axis=1)
        pred_LR_y = model_LR_server._model.eval(test_x)
        pred_LR_y = pred_LR_y.reshape(len(pred_LR_y), 1)
        pred_MLP_y = model_MLP_server._model.eval(test_x)
        pred_MLP_y = pred_MLP_y.reshape(len(pred_MLP_y), 1)
        server_LR_acc = np.count_nonzero(((pred_LR_y<0.5) + test_y)-1)/len(test_y)
        server_MLP_acc = np.count_nonzero(((pred_MLP_y<0.5) + test_y)-1)/len(test_y)
        server_acc = max(server_LR_acc, server_MLP_acc)
        
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
                             "server_predictor": ["both"],
                             "attacker_predictor": [args.attacker_predictor],
                             "min_auths": [args.min_auths],
                             "viable_CRPs": [len(challenges)],
                             "server_accuracy": [server_acc],
                             "attacker_accuracy": [attacker_acc]})
        data.to_csv(filepath, header=False, index=False, mode='a')
        
        # Train models
        model_LR_server.crps = ChallengeResponseSet(
            args.n_bits, challenges, responses)
        model_LR_server.fit()
        model_MLP_server.crps = ChallengeResponseSet(
            args.n_bits, challenges, responses)
        model_MLP_server.fit()
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

