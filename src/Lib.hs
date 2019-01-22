{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
-- {-# LANGUAGE DuplicateRecordFields #-}


module Lib
    ( someFunc
    ) where


import           GHC.Generics (Generic)

import           Control.Monad (forM)

import           System.Random (randomRIO)

-- import           Control.Lens.TH (makeLenses)
import           Lens.Micro.TH (makeLenses)
import           Lens.Micro ((^.))

import           Data.Maybe (fromJust)
import qualified Numeric.Backprop as BP
import           Numeric.Backprop ((^^.))
import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
import qualified Numeric.LinearAlgebra as LAD
import qualified Numeric.LinearAlgebra.Static as LA
import           Numeric.LinearAlgebra.Static.Backprop ((#>))
import qualified Debug.SimpleReflect as Refl


----------------------------------------------
-- Utils
----------------------------------------------


logistic :: Floating a => a -> a
logistic x = 1 / (1 + exp (-x))


reluSmooth :: Floating a => a -> a
reluSmooth x = log(1 + exp(x))


relu :: Floating a => a -> a
-- relu x = max 0 x
relu x = (x + abs x) / 2


-- myFunc :: Double -> Double
myFunc :: Floating a => a -> a
myFunc x = sqrt (x * 4)


----------------------------------------------
-- Net with dropout
----------------------------------------------


-- data DropNet = DropNet
--   { dropout1 :: LBP.R 100
--     -- ^ Input dropout: each element of the vector specifies the probability
--     -- of dropping the corresponding input unit
--   , weights1 :: LBP.L 20 100
--   , bias1    :: LBP.R 20
--   , dropout2 :: LBP.R 20
--     -- ^ Hidden layer dropout: each element of the vector specifies the
--     -- probability of dropping the corresponding hidden unit
--   , weights2 :: LBP.L 5 20
--   , bias2    :: LBP.R 5
--   }
--   deriving (Show, Generic)
--
--
-- runDropNet net x =
--   z
--   where
--     -- run input dropout
--     x' = x * dropout1 net
--     -- run hidden layer
--     y = logistic $ (weights1 net) LA.#> x' + (bias1 net)
--     -- run hidden dropout
--     y' = y * dropout2 net
--     -- run second layer
--     z = relu (weights2 net) LA.#> y' + (bias2 net)


----------------------------------------------
-- Net
----------------------------------------------


data Net = N
  { _nWeights1 :: LBP.L 20 100
  , _nBias1    :: LBP.R 20
  , _nWeights2 :: LBP.L 5 20
  , _nBias2    :: LBP.R 5
  }
  deriving (Show, Generic)

instance BP.Backprop Net

makeLenses ''Net


-- | Substract the second network from the first one.
subNet :: Net -> Net -> Double -> Net
subNet x y coef = N
  { _nWeights1 = _nWeights1 x - scale (_nWeights1 y)
  , _nBias1 = _nBias1 x - scale (_nBias1 y)
  , _nWeights2 = _nWeights2 x - scale (_nWeights2 y)
  , _nBias2 = _nBias2 x - scale (_nBias2 y)
  }
  where
    scale x
      = fromJust
      . LA.create
      . LAD.scale coef
      $ LA.unwrap x


data Dropout = Dropout
  { _saveInput :: LBP.R 100
    -- ^ Input dropout: each element of the vector specifies the probability
    -- of *preserving* the corresponding input unit
  , _saveHidden :: LBP.R 20
    -- ^ Hidden layer dropout: each element of the vector specifies the
    -- probability of *preserving* the corresponding hidden unit
  } deriving (Show, Generic)

instance BP.Backprop Dropout

makeLenses ''Dropout


runNet dp net x = z
  where
    -- run input dropout
    -- x' = x * (dp ^^. saveInput)
    x' = x
    -- run first layer
    y = logistic $ (net ^^. nWeights1) #> x' + (net ^^. nBias1)
    -- run hidden dropout
    -- y' = y * (dp ^^. saveHidden)
    y' = y
    -- run second layer
    z = relu (net ^^. nWeights2) #> y' + (net ^^. nBias2)


-- | Sample a concrete dropout with 0/1 values.
sample :: Dropout -> IO Dropout
sample dp = do
  dpIn <- doit (dp ^. saveInput)
  dpHd <- doit (dp ^. saveHidden)
  return $ Dropout
    { _saveInput = dpIn
    , _saveHidden = dpHd
    }
  where
    doit vect = fmap LA.vector . forM (toList vect) $ \p -> do
      x <- randomRIO (0, 1)
      return $ if x < p then 1 else 0
    toList = LAD.toList . LA.unwrap


----------------------------------------------
-- Error
----------------------------------------------


squaredError target output = error `LBP.dot` error
  where
    error = target - output


netError target input dropout net = squaredError
  (BP.auto target)
  (runNet (BP.auto dropout) net (BP.auto input))


-- squaredErrorDrop target output = error `LA.dot` error
--   where
--     error = target - output
--
--
-- netErrorDrop target input net = squaredErrorDrop
--   target
--   (runDropNet net input)


----------------------------------------------
-- Gradient Descent
----------------------------------------------


calcGrad target input dropout net =
  BP.gradBP (netError target input dropout) net


gradDesc
  :: Int
  -- ^ Number of iterations
  -> LA.R 100
  -- ^ Input vector
  -> LA.R 5
  -- ^ Target output vector
  -> Dropout
  -- ^ Dropout
  -> Net
  -- ^ Initial network
  -> IO Net
  -- ^ Resulting network
gradDesc iterNum input target dp net
  | iterNum > 0 = do
      print $ BP.evalBP (netError target input dp) net
      concDP <- sample dp
      let grad = calcGrad target input concDP net
          newNet = subNet net grad 0.3
      gradDesc (iterNum-1) input target dp newNet
  | otherwise = return net


----------------------------------------------
-- Main
----------------------------------------------


someFunc :: IO ()
someFunc = do
  print $ BP.evalBP myFunc (9 :: Double)
  print $ BP.gradBP myFunc (9 :: Double)
  print $ BP.evalBP myFunc (Refl.x :: Refl.Expr)
  print $ BP.gradBP myFunc (Refl.x :: Refl.Expr)

  -- an actual network
  let list k = take k $ cycle [0, 1, 2]
      zero k = take k $ repeat 0
      net = N
        { _nWeights1 = LA.matrix $ list (20*100)
        , _nBias1 = LA.vector $ list 20
        , _nWeights2 = LA.matrix $ list (5*20)
        , _nBias2 = LA.vector $ zero 5
        }
      dropout = Dropout
        { _saveInput = LA.vector . take 100 $ repeat 0.1
        , _saveHidden = LA.vector . take 20 $ repeat 0.1
        }
      input = LA.vector . take 100 $ repeat 1
      target = LA.vector [0.9, 1, 0.3, 0.4, 0.2]
      -- target = LA.vector [1, 2, 3, 4, 1]
      -- target = LA.vector [0.1, 0.0, 0.3, 0.4, 0.2]
  -- print myNet
  _net' <- gradDesc 100 input target dropout net

  return ()
