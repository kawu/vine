{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}


module Ex2 where


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
-- * the input vector of size `1` to
-- * the hidden vector of size `5` to
-- * the output vector of size `1`
data FFN = FFN
  { _nWeights1 :: L 5 1  -- first matrix
  , _nBias1    :: R 5    -- first bias
  , _nWeights2 :: L 2 5  -- second matrix
  , _nBias2    :: R 2    -- second bias
  }
  deriving (Show, Generic)

instance Backprop FFN
makeLenses ''FFN


-- evaluate the network `net` on the given input `x`
runFFN net x =
  -- take the first element of the resulting vector
  elem0 z
  where
    -- transform `x` to a singleton vector `v`
    v = vec1 x
    -- run first layer
    y = logistic ((net ^^. nWeights1) #> v + (net ^^. nBias1))
    -- run second layer
    z = relu ((net ^^. nWeights2) #> y + (net ^^. nBias2))


-- like `runFFN` but easier to use (because
-- not used for back-propagation)
evalFFN net x =
  evalBP2 runFFN net x


----------------------------------------------
-- Error
----------------------------------------------


-- squared error between `xs` and `ys`
squaredError xs ys = sum $ do
  (x, y) <- zip xs ys
  return $ (x-y)**2


-- error the network gives on our dataset
netError dataSet net =
  squaredError
    (map (runFFN net) input)
    target
  where
    input  = map (constVar . fst) dataSet
    target = map (constVar . snd) dataSet


----------------------------------------------
-- Training
----------------------------------------------


-- small dataset: data points to which we want to fit
-- our polynomial
trainData =
  [ (-2,  3)
  , ( 0,  2)
  , ( 1, -1)
  ]


-- train with the default dataset
train net =
  trainWith trainData net


-- train with a custom dataset
trainWith dataSet net =
  GD.gradDesc net (gdCfg dataSet)


-- main function
main = do
  net <- newFFN
  train net


-----------------------------------------------------
-- Gradient descent
-----------------------------------------------------


-- gradient descent configuration
gdCfg dataSet = GD.Config
  { iterNum = 5000
  , scaleCoef = 0.01
  , gradient = gradBP (netError dataSet)
  , substract = subFFN
  , quality = evalBP (netError dataSet)
  , reportEvery = 500
  }


-- | Substract the second network from the first one.
subFFN :: FFN -> FFN -> Double -> FFN
subFFN x y coef = 
  FFN
  { _nWeights1 = _nWeights1 x - scale coef (_nWeights1 y)
  , _nBias1 = _nBias1 x - scale coef (_nBias1 y)
  , _nWeights2 = _nWeights2 x - scale coef (_nWeights2 y)
  , _nBias2 = _nBias2 x - scale coef (_nBias2 y)
  }


-----------------------------------------------------
-- Random data construction
-----------------------------------------------------


-- create a new, random FFN
newFFN =
  FFN
    <$> matrix 5 1
    <*> vector 5
    <*> matrix 2 5
    <*> vector 2
