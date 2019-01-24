{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE TypeOperators #-}


module Ex1 where


import           GHC.Generics (Generic)

-- import           Prelude hiding (words)
-- 
-- import           GHC.TypeNats (KnownNat)
-- import qualified GHC.TypeNats as Nats
-- 
-- import           Control.Monad (forM, forM_)
-- 
-- import           System.Random (randomRIO)
-- 
import           Lens.Micro.TH (makeLenses)

-- import           Data.Maybe (fromJust)
-- import qualified Numeric.Backprop as BP
-- import           Numeric.Backprop ((^^.))
-- import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
-- import           Numeric.LinearAlgebra.Static.Backprop
--   (R, L, BVar, Reifies, W, (#))
-- import qualified Numeric.LinearAlgebra as LAD
-- import qualified Numeric.LinearAlgebra.Static as LA
-- import           Numeric.LinearAlgebra.Static.Backprop ((#>))
-- import qualified Debug.SimpleReflect as Refl
-- 
-- import           Basic
-- import qualified FeedForward as FFN
-- import           FeedForward (FFN(..))
-- import qualified GradientDescent as GD

import           Numeric.Backprop

import qualified GradientDescent as GD
import           GradientDescent (gradDesc, Config(..))


-----------------------------------------------------
-- Simple functions
-----------------------------------------------------


-- x to the power of 2
f x = x ** 2

-- exponential function
g x = exp x


-----------------------------------------------------
-- Polynomials
-----------------------------------------------------


-- polynomial a0 + a1*x + a2*x^2
data Poly = Poly
  { _a0 :: Double
  , _a1 :: Double
  , _a2 :: Double
  }
  deriving (Show, Generic)

instance Backprop Poly
makeLenses ''Poly


-- evaluate polynomial `p` on argument `x`
run p x =
  (p ^^. a0) +
  (p ^^. a1) * x +
  (p ^^. a2) * x^2


-- squared error between `xs` and `ys`
squaredError xs ys = sum $ do
  (x, y) <- zip xs ys
  return $ (x-y)**2


-- error the polynomial gives on our dataset
polyError dataSet poly =
  squaredError
    (map (run poly) input)
    target
  where
    input  = map (constVar . fst) dataSet
    target = map (constVar . snd) dataSet


-- small dataset: data points to which we want to fit
-- our polynomial
trainDefault =
  [ (-2,  3)
  , ( 0,  2)
  , ( 1, -1)
  -- , ( 2, -2)
  ]


-- train with the default dataset
train poly =
  trainWith poly trainDefault


-- train with a custom dataset
trainWith poly dataSet =
  GD.gradDesc poly (gdCfg dataSet)


-- main function
main = do
  train (Poly 0 0 0)


-----------------------------------------------------
-- Gradient descent
-----------------------------------------------------


-- gradient descent configuration
gdCfg dataSet = GD.Config
  { iterNum = 1000
  , scaleCoef = 0.01
  , gradient = gradBP (polyError dataSet)
  , substract = \x y k -> x `sub` (mul k y)
  , quality = evalBP (polyError dataSet)
  , reportEvery = 100
  }


-- substract the second poly params from the first
sub p1 p2 = Poly
  { _a0 = _a0 p1 - _a0 p2
  , _a1 = _a1 p1 - _a1 p2
  , _a2 = _a2 p1 - _a2 p2
  }


-- multiply the poly by `x`
mul x p = Poly
  { _a0 = x * _a0 p
  , _a1 = x * _a1 p
  , _a2 = x * _a2 p
  }
