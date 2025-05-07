#! /usr/bin/env python3

# Python distribution
from os import environ
from argparse import ArgumentParser

# Community
from   numpy import arange, empty, sqrt, round
from   numpy.random import default_rng
from   pandas import read_csv, DataFrame, concat
from   sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Local
from ESN import ESN
#from data_generator import generate_lorenz_data

# Silence: Warning: Ignoring XDG_SESSION_TYPE=wayland on Gnome
environ["XDG_SESSION_TYPE"] = "xcb"

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def main():
    '''Generation of data from ESN Reservoir Computer
       Compute RMSE of generated data

    Limitation: Since the outputs have to generate inputs in generative
    mode, the input/output variables muse be the same.
    This is wasteful for large problems. 

    If args.seed < 1 a random int is generated.

    Reservoir parameters
      R  = number of neurons in reservoir
      lr = leakage rate [0,1] (1.0 by default : no inertia )
           x(t+1) = (1-lr)*x(t) + lr*f[u(t+1),x(t)]
      sr = spectral radius of W
      rl = ridge regression lambda
    '''

    args = ParseCmdLine()

    if args.inFile :
        df   = read_csv( args.inFile )
        time = df.loc[ :, args.timeVar ].to_numpy()
    else :
        raise RuntimeError( 'inFile required' )

    N = len( args.variables  ) # number features

    # Indices for train, target = train + Tp
    train_i   = arange( start = args.train[0], stop = args.train[1] )
    target_i  = train_i + args.Tp
    eval_i    = arange( start = args.eval[0], stop = args.eval[1] )
    evalTgt_i = eval_i + args.Tp

    timeData = time[ train_i ]
    timeGen  = time[ eval_i ]
        
    df_ = df.loc[ :, args.variables ]

    # train, target data subsets
    trainData   = df_.iloc[train_i,  :].to_numpy()
    trainTarget = df_.iloc[target_i, :].to_numpy()
    evalData    = df_.iloc[eval_i,   :].to_numpy()
    evalTarget  = df_.iloc[evalTgt_i,:].to_numpy()

    # Instantiate ESN
    esn = ESN( num_inputs      = N,
               num_outputs     = N,
               num_resv_nodes  = args.Reservoir_N,
               leak_rate       = args.leakageRate,
               spectral_radius = args.spectralRadius,
               seed            = args.seed )

    # train network
    esn.train( trainData, trainTarget, args.ridgeLambda )

    # Generate 
    generated, mse_ = esn.predict_autonomous( evalData, evalTarget, args.burnIn )

    print( 'generated: ', generated.shape )

    # RMSE of testOut - generated
    error = evalTarget - generated
    print( 'error: ', error.shape )

    mse = empty( evalTarget.shape[0] )
    for i in range( evalTarget.shape[0] ) :
        mse[i] = mean_squared_error( evalTarget[i,:], generated[i,:] )
    rmse = round( sqrt( mse ), 4 )

    if args.outFile :
        df_esn = DataFrame( generated,
                            columns = [ f'{c}_' for c in args.variables ],
                            index = timeGen )
        df_data = DataFrame( evalTarget, columns = args.variables,
                             index = timeGen )

        dfOut = concat( [ df_data.round(4), df_esn.round(4) ], axis = 'columns' )
        dfOut['RMSE'] = rmse
        dfOut.index.rename( args.timeVar, inplace = True )
        dfOut.to_csv( args.outFile )

    #---------------------------------------------------------------------
    if args.plot :
        nPlots    = 4
        fig, axes = plt.subplots( nPlots, 1, figsize = (10,8), sharex = True )
        plt.suptitle( f"ESN {esn.num_resv_nodes} reservoir nodes",
                      fontsize = 14 )
        #labels   = args.variables

        columns = df.loc[ :, args.variables ].columns
        labels  = args.variables[-(nPlots-1):]
        label_i = [columns.get_loc(c) for c in labels ]

        if args.xlim :
            plt.xlim( args.xlim[0], args.xlim[1] )

        #for i, label in enumerate( labels ):
        for i in range( nPlots-1 ) :
            label = labels[i]
            i_    = label_i[i]

            ax = axes[i]
            #ax.plot(timeData, evalTarget[:, i_], label = "Data",
            #        color = 'steelblue', lw = 3)
            ax.plot(time, df.loc[:, label], label = label,
                    color = 'steelblue', lw = 3)
            ax.plot(timeGen, generated[:, i_], label = "ESN",
                    color = 'darkorange', linestyle = '--', lw = 3)

            ax.axvline( x = time[ args.train[1] + args.burnIn ],
                        color="red", lw = 2 )
            ax.set_ylabel( label, fontsize = 12 )
            ax.legend( loc = "lower left", fontsize = 12 )
            ax.tick_params( axis = 'both', labelsize = 12 )

        ax = axes[3]
        ax.plot(timeGen, rmse, label = "RMSE", color = 'brown', lw = 3)
        ax.axvline( x = time[ args.train[1] + args.burnIn ],
                    color="red", lw = 2 )
        ax.set_ylabel( "RMSE", fontsize = 12 )
        ax.tick_params( axis='both', labelsize = 12 )
        ax.set_xlabel( args.timeVar, fontsize = 12 )

        plt.tight_layout()
        plt.show()

#----------------------------------------------------------------------------
def ParseCmdLine():
    '''If seed < 1, generate one'''

    parser = ArgumentParser( description = 'ESN Lorenz' )

    parser.add_argument('-i', '--inFile',
                        dest   = 'inFile', type = str, 
                        action = 'store',  default = 'data/Lorenz3D_4k.csv',
                        help = '.csv data file')

    parser.add_argument('-o', '--outFile',
                        dest   = 'outFile', type = str, 
                        action = 'store',   default = None,
                        help = '.csv data file')

    parser.add_argument('-iv', '--variables', nargs = '*',
                        dest   = 'variables',  type = str, 
                        action = 'store',      default = ['V1', 'V2', 'V3'],
                        help = 'input variable names')

    parser.add_argument('-time', '--timeVar',
                        dest   = 'timeVar',  type = str, 
                        action = 'store',    default = 'Time',
                        help = '.csv file time label')

    parser.add_argument('-Tp', '--Tp',
                        dest   = 'Tp',     type = int, 
                        action = 'store',  default = 1,
                        help = 'prediction horizon time steps')

    parser.add_argument('-R', '--Reservoir_N',
                        dest   = 'Reservoir_N', type = int, 
                        action = 'store',       default = 1000,
                        help = 'number of neurons in reservoir')

    parser.add_argument('-lr', '--leakageRate',
                        dest   = 'leakageRate', type = float, 
                        action = 'store',       default = 1.,
                        help = 'leakage rate')

    parser.add_argument('-rl', '--ridgeLambda',
                        dest   = 'ridgeLambda', type = float, 
                        action = 'store',  default = 0.0001,
                        help = 'Ridge regression lambda')

    parser.add_argument('-sr', '--spectralRadius',
                        dest   = 'spectralRadius', type = float, 
                        action = 'store',          default = 1.0,
                        help = 'spectral Radius')

    parser.add_argument('-b', '--burnIn',
                        dest   = 'burnIn', type = int, 
                        action = 'store',  default = 100,
                        help = 'generative mode burn-in time steps')

    parser.add_argument('-t', '--train', nargs = 2,
                        dest   = 'train', type = int, 
                        action = 'store', default = [1,1900],
                        help = 'train indices')

    parser.add_argument('-e', '--eval', nargs = 2,
                        dest   = 'eval', type = int, 
                        action = 'store', default = [2001,2500],
                        help = 'eval indices')

    parser.add_argument('-s', '--seed',
                        dest   = 'seed', type = int, 
                        action = 'store', default = 765432,
                        help = 'seed')

    parser.add_argument('-x', '--xlim',   nargs = 2,
                        dest   = 'xlim',  type = float,
                        action = 'store', default = None,
                        help = 'plot xlim')

    parser.add_argument('-P', '--plot',
                        dest   = 'plot', 
                        action = 'store_true', default = False,
                        help = 'plot')

    args = parser.parse_args()

    if args.seed < 1 :
        rng       = default_rng()
        args.seed = rng.integers( 0, 9E18 ) # int64 max is 9.2E18
        print( 'Random seed set to ', args.seed )

    if args.spectralRadius < 1E-5 :
        raise RuntimeError( 'ParseCmdLine: spectral radius too small' )

    return args

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading
if __name__ == "__main__":
    main()
