{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}


module Ex3 where


import           GHC.Generics (Generic)

import           Lens.Micro.TH (makeLenses)

import qualified Numeric.LinearAlgebra.Static as LA
import qualified Numeric.LinearAlgebra        as LAD

import           Numeric.Backprop
import           Numeric.LinearAlgebra.Static.Backprop
                   (R, L, (#), (#>), dot)

import qualified GradientDescent as GD
import           Basic


----------------------------------------------
-- Feed-forward net
----------------------------------------------


-- A feed-forward network with one hidden layer.
-- It transforms:
-- * the input vector of size `10` to
-- * the hidden vector of size `5` to
-- * the output vector of size `5`
data FFN = FFN
  { _nWeights1 :: L 5 10 -- first matrix
  , _nBias1    :: R 5    -- first bias
  , _nWeights2 :: L 5 5  -- second matrix
  , _nBias2    :: R 5    -- second bias
  }
  deriving (Show, Generic)

instance Backprop FFN
makeLenses ''FFN


-- evaluate the network `net` on the given input vector `x`
runFFN net x = z
  where
    -- run first layer
    y = logistic ((net ^^. nWeights1) #> x + (net ^^. nBias1))
    -- run second layer
    z = relu ((net ^^. nWeights2) #> y + (net ^^. nBias2))


----------------------------------------------
-- RNN
----------------------------------------------


-- A recursive neural network which transforms:
-- * the sequence of input vectors, each of size `5`, to
-- * the sequence of hidden vector, each of size `5`
data RNN = RNN
  { _ffn :: FFN
    -- ^ The underlying network used to calculate subsequent hidden values
  , _h0  :: R 5
    -- ^ The initial hidden state
  }
  deriving (Show, Generic)

instance Backprop RNN
makeLenses ''RNN


-- run the recursive neural network
-- on the given list of input vectors
runRNN net []     = net ^^. h0
runRNN net (x:xs) = h
  where
    -- run the recursive calculation
    h' = runRNN net xs
    -- calculate the new hidden value
    h = runFFN (net ^^. ffn) (x # h')


-- like `runRNN` but easier to use (because
-- not used for back-propagation)
evalRNN net xs =
  LAD.toList (LA.unwrap h)
  where
    h = evalBP0
          ( runRNN
              (constVar net)
              (map constVar xs)
          )


----------------------------------------------
-- Error
----------------------------------------------


-- squared error between two vectors
squaredError1 target output =
  err `dot` err
  where
    err = target - output


-- a sum of squared errors for each (target, output) pair
squaredError targets outputs =
  sum $ do
    (target, output) <- zip targets outputs 
    return $ squaredError1 target output


-- calculate the error of the predictions of the network
netError dataSet net =
  let
    inputs = map fst dataSet
    outputs = map (runRNN net . map constVar) inputs
    targets = map (auto . snd) dataSet
  in  
    squaredError targets outputs


----------------------------------------------
-- Training
----------------------------------------------


-- simple vectors: only ones or only zeros
ones = LA.vector [1, 1, 1, 1, 1]
zers = LA.vector [0, 0, 0, 0, 0]


-- IDEA: teach the network to output `zers` if all
-- input vectors are `zers` and `ones` otherwise


-- the training dataset consists of pairs
-- (input, target output)
trainData =
  [ ([zers, zers, zers, zers, zers], zers)
  , ([zers, zers, zers, zers, ones], ones)
  , ([zers, zers, zers, ones, zers], ones)
  , ([zers, zers, ones, zers, zers], ones)
  , ([zers, ones, zers, zers, ones], ones)
  , ([ones, zers, zers, zers, zers], ones)
  , ([ones, ones, ones, ones, ones], ones)
  ]


-- alternative training data
trainData2 =
  [ ([], ones)
  , ([ones], ones)
  , ([zers], zers)
  , ([ones, ones], ones)
  , ([zers, ones], zers)
  , ([ones, zers], zers)
  , ([zers, zers], zers)
  ] 


-- train with the default dataset
train net =
  trainWith trainData net


-- train with a custom dataset
trainWith dataSet net =
  GD.gradDesc net (gdCfg dataSet)


-- main function
main = do
  net <- newRNN
  train net


-----------------------------------------------------
-- Gradient descent
-----------------------------------------------------


-- gradient descent configuration
gdCfg dataSet = GD.Config
  { iterNum = 2500
  , scaleCoef = 0.01
  , gradient = gradBP (netError dataSet)
  , substract = subRNN
  , quality = evalBP (netError dataSet)
  , reportEvery = 100
  }


-- | Substract the second network from the first one.
subFFN :: FFN -> FFN -> Double -> FFN
subFFN x y coef = FFN
  { _nWeights1 = _nWeights1 x - scale coef (_nWeights1 y)
  , _nBias1 = _nBias1 x - scale coef (_nBias1 y)
  , _nWeights2 = _nWeights2 x - scale coef (_nWeights2 y)
  , _nBias2 = _nBias2 x - scale coef (_nBias2 y)
  }


-- | Substract the second network from the first one.
subRNN :: RNN -> RNN -> Double -> RNN
subRNN x y coef = RNN
  { _ffn = subFFN (_ffn x) (_ffn y) coef
  , _h0 = _h0 x - scale coef (_h0 y)
  }


-----------------------------------------------------
-- Random data construction
-----------------------------------------------------


-- create a new, random FFN
newFFN =
  FFN
    <$> matrix 5 10
    <*> vector 5
    <*> matrix 5 5
    <*> vector 5


-- create a new, random RRN
newRNN =
  RNN
    <$> newFFN
    <*> vector 5
