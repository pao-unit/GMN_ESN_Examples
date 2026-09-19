# Distribution
from   typing import Callable, Tuple

# Community
import numpy             as np
from   scipy             import linalg
from   scipy.sparse      import csr_matrix
from   scipy.sparse.linalg import eigs as SparseEigs

#--------------------------------------------------------------------------
class InputLayer:
#--------------------------------------------------------------------------
    """Input layer W_in applied to the augmented input [1 ; u].

       Row 0 of W is the bias column : this is the Jaeger W_in [1;u]
       formulation, giving every reservoir node an independent random
       bias rather than one scalar shared by all nodes.
    """

    def __init__( self, numInputs : int, numNodes : int,
                  inputScale : float = 1.0, inputBias : float = 1.0,
                  rng = None ) -> None:
        self.numInputs  = numInputs
        self.numNodes   = numNodes
        self.inputScale = inputScale
        self.inputBias  = inputBias
        self.rng        = rng if rng is not None else np.random.default_rng()
        self.W          = None

        self.InitWeights()

    def InitWeights( self ) -> None:
        # uniform distribution [-1, 1] over the augmented input [1 ; u]
        W = self.rng.uniform( -1, 1, size = ( self.numInputs + 1,
                                              self.numNodes ) )
        W[0,  :] *= self.inputBias   # per-node bias column
        W[1:, :] *= self.inputScale  # input weights

        self.W = W

    def Reset( self ) -> None:
        self.InitWeights()

    def __call__( self, x : np.ndarray ) -> np.ndarray:
        return np.dot( x, self.W[1:, :] ) + self.W[0, :]

#--------------------------------------------------------------------------
class ReservoirLayer:
#--------------------------------------------------------------------------
    """Sparse leaky-integrator reservoir.

       The state update is r W, so column i of W holds the in-coming
       connections of node i.  `degree` non-zeros are placed in each
       COLUMN, giving every node exactly `degree` inputs and making it
       impossible to orphan a node.  Self connections are permitted.
    """

    def __init__( self, numNodes : int, leakRate : float,
                  spectralRadius : float, degree : int,
                  activation : Callable = np.tanh, rng = None ) -> None:
        self.numNodes       = numNodes
        self.leakRate       = leakRate
        self.spectralRadius = spectralRadius  # target, never overwritten
        self.degree         = degree
        self.activation     = activation
        self.rng            = rng if rng is not None \
                              else np.random.default_rng()
        self.W              = None
        self.Wt             = None
        self.radiusInit     = None  # spectral radius before scaling
        self.radiusFinal    = None  # spectral radius after  scaling

        self.InitWeights()

    def MaxAbsEigenvalue( self, W ) -> float:
        """Largest |eigenvalue| of W.

           ARPACK can fail to converge on a sparse, strongly non-normal
           random matrix : fall back to the dense solver rather than
           return a silently wrong spectral radius.
        """
        if self.numNodes > 3 :
            try :
                eigenValues = SparseEigs( W.astype( float ), k = 1,
                                          which = 'LM',
                                          return_eigenvectors = False )
                return float( abs( eigenValues[0] ) )
            except Exception :
                print( 'ESN:ReservoirLayer.MaxAbsEigenvalue : ' +\
                       'sparse solver failed, using dense solver' )

        return float( max( abs( linalg.eigvals( W.toarray() ) ) ) )

    def InitWeights( self ) -> None:
        numNodes = self.numNodes
        degree   = self.degree

        rowInd = np.empty( numNodes * degree, dtype = int )
        colInd = np.empty( numNodes * degree, dtype = int )

        for i in range( numNodes ):
            j = i * degree
            rowInd[ j : j + degree ] = self.rng.choice( numNodes,
                                                        size    = degree,
                                                        replace = False )
            colInd[ j : j + degree ] = i

        values = self.rng.normal( 0, 1, numNodes * degree )

        W = csr_matrix( ( values, ( rowInd, colInd ) ),
                        shape = ( numNodes, numNodes ) )

        # Initial spectral radius : computed after masking since
        # zeroing entries changes the eigenvalues
        self.radiusInit = self.MaxAbsEigenvalue( W )

        if self.radiusInit < 1E-12 :
            raise RuntimeError( 'ESN:ReservoirLayer.InitWeights : ' +\
                                'degenerate reservoir, spectral radius 0' )

        # Scale to the requested spectral radius (eigenvalues are linear op)
        self.W  = csr_matrix( W * ( self.spectralRadius / self.radiusInit ) )
        self.Wt = csr_matrix( self.W.T )

        self.radiusFinal = self.MaxAbsEigenvalue( self.W )

    def Reset( self ) -> None:
        self.InitWeights()

    def __call__( self, rTime : np.ndarray,
                  uTime : np.ndarray ) -> np.ndarray:
        # Wt.dot( r ) is ( r W ) for the row-vector state convention
        return ( 1 - self.leakRate ) * rTime + self.leakRate * \
            self.activation( self.Wt.dot( rTime ) + uTime )

#--------------------------------------------------------------------------
class OutputLayer:
#--------------------------------------------------------------------------
    """Linear readout over the assembled feature vector [1 ; u ; r].

       There is no separate bias term : the intercept is the constant
       feature and is fit by the ridge regression along with the rest.
    """

    def __init__( self, numFeatures : int, numOutputs : int ) -> None:
        self.numFeatures = numFeatures
        self.numOutputs  = numOutputs
        self.W           = None

        self.InitWeights()

    def InitWeights( self ) -> None:
        # untrained readout produces zero, it is solved for analytically
        self.W = np.zeros( ( self.numFeatures, self.numOutputs ) )

    def Reset( self ) -> None:
        self.InitWeights()

    def UpdateWeights( self, weights : np.ndarray ) -> None:
        if weights.shape != ( self.numFeatures, self.numOutputs ) :
            raise RuntimeError( 'ESN:OutputLayer.UpdateWeights : ' +\
                                f'weights {weights.shape} != ' +\
                                f'{( self.numFeatures, self.numOutputs )}' )
        self.W = weights

    def __call__( self, features : np.ndarray ) -> np.ndarray:
        return np.dot( features, self.W )

#--------------------------------------------------------------------------
class ESN:
#--------------------------------------------------------------------------
    """Echo State Network using Reservoir Computing"""

    def __init__(
        self,
        numInputs      : int,
        numOutputs     : int,
        numResvNodes   : int,
        leakRate       : float = 0.5,
        spectralRadius : float = 0.9,
        degree         : int   = 5,
        activation     : Callable = np.tanh,
        inputScale     : float = 1.0,
        inputBias      : float = 1.0,
        readoutBias    : bool  = True,
        readoutInput   : bool  = False,
        warmStart      : bool  = True,
        seed           : int   = 0
    ) -> None:
        """

        Parameters:
            numInputs (int): number of units for the input layer
            numOutputs (int): number of units for the output layer
            numResvNodes (int): number of reservoir nodes
            leakRate (float): the leaky rate for reservoir layer
            spectralRadius (float): spectral radius for reservoir layer
            degree (int): in-coming connections per reservoir node
            activation (callable): activation function for reservoir nodes
            inputScale (float): scaling of the input weights
            inputBias (float): scaling of the per-node random bias column
            readoutBias (bool): fit an intercept in the readout
            readoutInput (bool): include direct input -> output connections.
                Off by default : in generative mode this is a feedthrough
                path bypassing the reservoir memory, which helps one-step
                prediction but can destabilise long autonomous rollouts.
            warmStart (bool): begin prediction from the final training
                reservoir state instead of zeros.  Only correct when the
                prediction window is contiguous with the training window.
            seed (int): RNG seed

        Note : spectral radius is defined as the largest absolute eigenvalue
               of the reservoir weight matrix W.

        Code adapted from:
            ReservoirComputing: Implementing Reservoir Computing Networks
            for Predicting Dynamic Systems}, Michael Hu,
            https://github.com/michaelnny/ReservoirComputing,
            version = 1.0.0, 2023
        """

        if numInputs < 1 :
            raise RuntimeError( 'ESN.__init__ : numInputs < 1' )
        if numOutputs < 1 :
            raise RuntimeError( 'ESN.__init__ : numOutputs < 1' )
        if numResvNodes < 1 :
            raise RuntimeError( 'ESN.__init__ : numResvNodes < 1' )
        if not 0 < leakRate <= 1 :
            raise RuntimeError( 'ESN.__init__ : leakRate not in (0,1]' )
        if spectralRadius <= 0 :
            raise RuntimeError( 'ESN.__init__ : spectralRadius <= 0' )
        if not 1 <= degree < numResvNodes :
            raise RuntimeError( 'ESN.__init__ : degree not in ' +\
                                f'[1,{numResvNodes})' )
        if inputScale < 0 :
            raise RuntimeError( 'ESN.__init__ : inputScale < 0' )
        if inputBias < 0 :
            raise RuntimeError( 'ESN.__init__ : inputBias < 0' )

        self.numInputs      = numInputs
        self.numOutputs     = numOutputs
        self.numResvNodes   = numResvNodes
        self.leakRate       = leakRate
        self.spectralRadius = spectralRadius
        self.degree         = degree
        self.inputScale     = inputScale
        self.inputBias      = inputBias
        self.readoutBias    = readoutBias
        self.readoutInput   = readoutInput
        self.warmStart      = warmStart
        self.seed           = seed

        self.numFeatures = numResvNodes + \
                           ( 1 if readoutBias  else 0 ) + \
                           ( numInputs if readoutInput else 0 )

        # per-instance Generator : no global np.random side effects
        self.rng = np.random.default_rng( seed )

        self.inputLayer = InputLayer( numInputs, numResvNodes,
                                      inputScale, inputBias, self.rng )

        self.resvLayer = ReservoirLayer( numResvNodes, leakRate,
                                         spectralRadius, degree,
                                         activation, self.rng )

        self.outputLayer = OutputLayer( self.numFeatures, numOutputs )

        self.resvStates     = None
        self.resvStateFinal = None
        self.trained        = False

        self.PrintStatus()

    #---------------------------------------------------------------------
    def PrintStatus( self ) -> None:
        """Unconditional status of the network configuration at init"""
        density = 100.0 * self.degree / self.numResvNodes

        print( 'ESN.__init__ : Echo State Network' )
        print( f'  seed           = {self.seed}' )
        print( f'  numInputs      = {self.numInputs}' )
        print( f'  numOutputs     = {self.numOutputs}' )
        print( f'  numResvNodes   = {self.numResvNodes}' )
        print( f'  degree         = {self.degree}' +\
               f'   (density {density:.3f}%)' )
        print( f'  leakRate       = {self.leakRate}' )
        print( f'  spectralRadius = {self.spectralRadius}' +\
               f'   (achieved {self.resvLayer.radiusFinal:.6f})' )
        print( f'  inputScale     = {self.inputScale}' )
        print( f'  inputBias      = {self.inputBias}' )
        print( f'  readoutBias    = {self.readoutBias}' )
        print( f'  readoutInput   = {self.readoutInput}' )
        print( f'  warmStart      = {self.warmStart}' )
        print( f'  numFeatures    = {self.numFeatures}', flush = True )

        if self.spectralRadius > 1.0 + 1E-6 :
            print( 'ESN.__init__ : WARNING spectralRadius ' +\
                   f'{self.spectralRadius} > 1 : the echo state property ' +\
                   'is not assured', flush = True )

    #---------------------------------------------------------------------
    def Reset( self ) -> None:
        # restore the Generator so a Reset network is reproducible
        self.rng = np.random.default_rng( self.seed )

        self.inputLayer.rng = self.rng
        self.resvLayer.rng  = self.rng

        self.inputLayer.Reset()
        self.resvLayer.Reset()
        self.outputLayer.Reset()

        self.resvStates     = None
        self.resvStateFinal = None
        self.trained        = False

    #---------------------------------------------------------------------
    def InitialState( self ) -> np.ndarray:
        """Reservoir state to begin a prediction from.

           A copy is returned so that prediction never mutates the stored
           training state : repeated calls are reproducible.
        """
        if self.warmStart :
            if self.resvStateFinal is not None :
                return self.resvStateFinal.copy()

            print( 'ESN.InitialState : WARNING warmStart requested but ' +\
                   'the network is not trained : starting from zeros',
                   flush = True )

        return np.zeros( self.numResvNodes )

    #---------------------------------------------------------------------
    def ReadoutFeatures( self, resvState : np.ndarray,
                         inputVec : np.ndarray ) -> np.ndarray:
        """Assemble the readout design vector/matrix [1 ; u ; r]"""
        oneD = resvState.ndim == 1

        r = resvState.reshape( 1, -1 ) if oneD else resvState
        u = inputVec.reshape( 1, -1 )  if oneD else inputVec

        parts = []
        if self.readoutBias :
            parts.append( np.ones( ( r.shape[0], 1 ) ) )
        if self.readoutInput :
            parts.append( u )
        parts.append( r )

        features = np.hstack( parts )

        return features[0] if oneD else features

    #---------------------------------------------------------------------
    def Train(
        self,
        trainInput  : np.ndarray,
        trainTarget : np.ndarray,
        ridgeLambda : float = 0.1,
        washout     : int   = None,
    ) -> None:
        """
        Train the model by running through the input data and collecting
        reservoir states, then update the weights of the output layer.

        Parameters:
            trainInput (np.ndarray): input data shape of [sequenceLen, N],
                where N is the number of features fed to the input layer.
            trainTarget (np.ndarray): target data shape of [sequenceLen, M],
                where M is the number of features to predict.
            ridgeLambda (float, optional): lambda for the Ridge Regression
                (default 0.1).
            washout (int, optional): initial reservoir states discarded
                before the regression.  These reflect the arbitrary zero
                initial state rather than the input history.  None selects
                min( 100, sequenceLen // 10 ).
        """

        if len( trainInput ) != len( trainTarget ) :
            raise RuntimeError( 'ESN.Train : trainInput and trainTarget ' +\
                                'lengths differ' )
        if len( trainInput.shape ) != 2 or len( trainTarget.shape ) != 2 :
            raise RuntimeError( 'ESN.Train : input and target must be 2-D' )

        T = len( trainInput )

        if washout is None :
            washout = min( 100, T // 10 )
        if not 0 <= washout < T :
            raise RuntimeError( f'ESN.Train : washout {washout} not in ' +\
                                f'[0,{T})' )

        resvStates = np.empty( ( T, self.numResvNodes ) )

        # initialize dummy reservoir state for first timestep
        rTime = np.zeros( self.numResvNodes )

        # collect reservoir states
        for t in range( T ):
            uTime = self.inputLayer( trainInput[t] )
            rTime = self.resvLayer( rTime, uTime )
            resvStates[t, :] = rTime

        self.resvStates     = resvStates
        self.resvStateFinal = rTime.copy()

        # discard the washout transient, then update output layer weights
        features = self.ReadoutFeatures( resvStates[ washout :, : ],
                                         trainInput[ washout :, : ] )

        self.UpdateOutputWeights( features, trainTarget[ washout :, : ],
                                  ridgeLambda )
        self.trained = True

        print( f'ESN.Train : {T} steps, washout {washout}, ' +\
               f'ridgeLambda {ridgeLambda}, ' +\
               f'{features.shape[1]} readout features', flush = True )

    #---------------------------------------------------------------------
    def UpdateOutputWeights( self, features : np.ndarray,
                             target : np.ndarray,
                             ridgeLambda : float ) -> None:
        """Compute the output weights analytically"""
        # Ridge Regression
        numFeatures = features.shape[1]
        E_lambda    = np.identity( numFeatures ) * ridgeLambda

        # the intercept is not penalised
        if self.readoutBias :
            E_lambda[0, 0] = 0.0

        A = np.dot( features.T, features ) + E_lambda
        B = np.dot( features.T, target )

        # solve the normal equations rather than form an explicit inverse
        try :
            outWeights = linalg.solve( A, B, assume_a = 'sym' )
        except Exception :
            print( 'ESN.UpdateOutputWeights : singular system, ' +\
                   'using least squares' )
            outWeights = linalg.lstsq( A, B )[0]

        outWeights = outWeights.reshape( self.outputLayer.W.shape )

        self.outputLayer.UpdateWeights( outWeights )

    #---------------------------------------------------------------------
    def Predict( self, inputData : np.ndarray,
                 trueTarget : np.ndarray ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use the trained model to perform teacher-forced prediction

        Parameters:
            inputData (numpy.ndarray): input data shape of [sequenceLen, N],
                where N is the number of features fed to the input layer
            trueTarget (numpy.ndarray): target data shape of [sequenceLen, M],
                where M is the number of features to predict

        Returns:
            tuple:
                predTarget (numpy.ndarray): a 2D numpy.ndarray with
                shape of [sequenceLen, M] contains the predicted values
                mse (numpy.ndarray): a 1D numpy.ndarray with shape of [M]
                contains the MSE for each predicted dimension
        """
        if len( inputData ) != len( trueTarget ) :
            raise RuntimeError( 'ESN.Predict : inputData and trueTarget ' +\
                                'lengths differ' )
        if len( inputData.shape ) != 2 or len( trueTarget.shape ) != 2 :
            raise RuntimeError( 'ESN.Predict : input and target must be 2-D' )

        T          = len( inputData )
        predTarget = np.empty( ( T, self.numOutputs ) )

        warm  = self.warmStart and self.resvStateFinal is not None
        rTime = self.InitialState()

        for t in range( T ):
            uTime = self.inputLayer( inputData[t] )
            rTime = self.resvLayer( rTime, uTime )

            features = self.ReadoutFeatures( rTime, inputData[t] )

            predTarget[t, :] = self.outputLayer( features )

        # a warm started reservoir has no dummy state to skip
        i0 = 0 if warm else 1

        squaredDiff = ( predTarget[i0:] - trueTarget[i0:] ) ** 2
        mse         = np.mean( squaredDiff, axis = 0 )

        return predTarget, mse

    #---------------------------------------------------------------------
    def PredictAutonomous( self, inputData : np.ndarray,
        trueTarget : np.ndarray,
        burnIn : int = 1 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use the trained model to perform autonomous prediction.

        Args:
            inputData (numpy.ndarray): input data shape of [sequenceLen, N],
                where N is the number of features fed to the input layer.
            trueTarget (numpy.ndarray): target data shape of [sequenceLen, N],
                where N is the number of features to predict.
            burnIn (int, optional): number of timesteps driven by real data
                from inputData before the model free runs on its own
                predictions (default 1).

        Returns:
            tuple:
                predTarget (numpy.ndarray): a 2D numpy.ndarray with shape
                of [sequenceLen, M] contains the predicted values.  The
                full span is returned, burnIn steps included.
                mse (numpy.ndarray): a 1D numpy.ndarray with shape of [M]
                contains the MSE for each predicted dimension, computed
                only over the free running steps after burnIn.
        """

        if len( inputData.shape ) != 2 or len( trueTarget.shape ) != 2 :
            raise RuntimeError( 'ESN.PredictAutonomous : input and target ' +\
                                'must be 2-D' )
        if self.numInputs != self.numOutputs :
            raise RuntimeError( 'ESN.PredictAutonomous : requires ' +\
                                'numInputs == numOutputs to feed back' )

        T = len( trueTarget )

        if not 0 <= burnIn < T :
            raise RuntimeError( f'ESN.PredictAutonomous : burnIn {burnIn} ' +\
                                f'not in [0,{T})' )

        predTarget = np.empty( ( T, self.numOutputs ) )

        rTime = self.InitialState()
        xTime = inputData[0, :]

        for t in range( T ):
            uTime = self.inputLayer( xTime )
            rTime = self.resvLayer( rTime, uTime )

            features = self.ReadoutFeatures( rTime, xTime )
            xNext    = self.outputLayer( features )

            predTarget[t, :] = xNext

            if t >= burnIn :
                xTime = xNext
            elif t + 1 < len( inputData ) :
                # teacher forcing : advance to the next observation
                xTime = inputData[ t + 1, : ]
            else :
                xTime = xNext

        # MSE over the free running steps only : the burnIn steps are
        # driven by real data and do not measure predictive skill
        squaredDiff = ( predTarget[burnIn:] - trueTarget[burnIn:] ) ** 2
        mse         = np.mean( squaredDiff, axis = 0 )

        return predTarget, mse
