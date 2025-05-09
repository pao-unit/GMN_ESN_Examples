#! /usr/bin/env python3

# Python distribution
from os import environ
from argparse import ArgumentParser

# Community
from   numpy import arange, empty, sqrt, round, mean
from   pandas import read_csv, DataFrame, concat
#from   sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Local

# Silence: Warning: Ignoring XDG_SESSION_TYPE=wayland on Gnome
environ["XDG_SESSION_TYPE"] = "xcb"

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def main():
    '''Plot Lorenz '63 x,y,z
       Compute RMSE of generated x,y,z
    '''

    args = ParseCmdLine()

    dfGen  = read_csv( args.inFile )
    dfData = read_csv( args.dataFile )

    # Set timeVar as index to get data subset according to gen
    dfGen.set_index ( args.timeVar, drop = True, inplace = True )
    dfData.set_index( args.timeVar, drop = True, inplace = True )

    # subset dfData to same index as dfGen
    dfDataGen = dfData.loc[ dfGen.index, args.dataVariables ]

    #----------------------------------------------------------------
    # RMSE of data - generated
    # Convert to numpy for error difference (pandas matches column
    # names in DataFrame difference) then back to DataFrame
    error = dfDataGen.loc[:, args.dataVariables].to_numpy() - \
            dfGen.loc    [:, args.genVariables].to_numpy()
    error = DataFrame( error, columns = args.dataVariables )

    #print( error.head(2) )

    def MeanSquare( x ) :
        return mean( x * x )
    mse  = error.apply( MeanSquare, axis = 'columns' )
    rmse = round( sqrt( mse ), 4 )

    #----------------------------------------------------------------
    fig, axes = plt.subplots( 4, 1, figsize = (8,6.5), sharex = True )
    plt.suptitle( args.title, fontsize = 14 )
    timeData = dfData.index.to_numpy()
    timeGen  = dfGen.index.to_numpy()

    if args.xlim :
        plt.xlim( args.xlim[0], args.xlim[1] )

    for i in range( len( args.dataVariables ) ) :
        dataLabel = args.dataVariables[i]
        genLabel  = args.genVariables[i]
        ax        = axes[i]

        ax.plot(timeData, dfData.loc[:, dataLabel], label = "Data",
                color = 'steelblue', lw = 3)
        ax.plot(timeGen, dfGen.loc[:, genLabel], label = args.generator,
                color = 'darkorange', linestyle = '--', lw = 3)
        ax.axvline( x = args.genStart, color="red", lw = 2 )
        ax.set_ylabel( dataLabel, fontsize = 12 )
        if i == 0 :
            ax.legend( loc = "upper left", fontsize = 12 )
        ax.tick_params( axis = 'both', labelsize = 12 )

    ax = axes[3]
    ax.plot(timeGen, rmse, label = "RMSE", color = 'brown', lw = 3)
    ax.axvline( x = args.genStart, color="red", lw = 2 )
    ax.set_ylabel( "RMSE", fontsize = 12 )
    ax.tick_params( axis='both', labelsize = 12 )
    ax.set_xlabel( args.timeVar, fontsize = 12 )

    plt.tight_layout()
    plt.show()

#----------------------------------------------------------------------------
def ParseCmdLine():
    ''' '''

    parser = ArgumentParser( description = 'Plot GMN' )

    parser.add_argument('-d', '--dataFile',
                        dest   = 'dataFile', type = str, 
                        action = 'store',
                        default = '../data/Lorenz3D_4k.csv',
                        help = '.csv data file')

    parser.add_argument('-i', '--inFile',
                        dest   = 'inFile', type = str, 
                        action = 'store',
                        default = 'GMN_E3_tau-7_pS_2000_pL_1000.csv',
                        help = '.csv GMN out file')

    parser.add_argument('-g', '--generator',
                        dest   = 'generator', type = str, 
                        action = 'store',     default = 'GMN',
                        help = 'algorithm name')

    parser.add_argument('-dv', '--dataVariables', nargs = '*',
                        dest   = 'dataVariables', type = str, 
                        action = 'store',         default = ['V1', 'V2', 'V3'],
                        help = 'data variable names')

    parser.add_argument('-gv', '--genVariables', nargs = '*',
                        dest   = 'genVariables', type = str, 
                        action = 'store',        default = ['V1', 'V2', 'V3'],
                        help = 'generated variable names')

    parser.add_argument('-time', '--timeVar',
                        dest   = 'timeVar',  type = str, 
                        action = 'store',    default = 'Time',
                        help = '.csv file time label')

    parser.add_argument('-gs', '--genStart',
                        dest   = 'genStart',  type = float, 
                        action = 'store',     default = 50.02,
                        help = 'start time of generated data')

    parser.add_argument('-t', '--title',
                        dest   = 'title', type = str, 
                        action = 'store', default = 'GMN [V1,V2] -> V3',
                        help = 'plot title')

    parser.add_argument('-x', '--xlim',   nargs = 2,
                        dest   = 'xlim',  type = float,
                        action = 'store', default = [48,61],
                        help = 'plot xlim')

    args = parser.parse_args()

    return args

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading
if __name__ == "__main__":
    main()
