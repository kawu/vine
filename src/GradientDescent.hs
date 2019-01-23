{-# LANGUAGE RecordWildCards #-}


module GradientDescent
  ( Config (..)
  , gradDesc
  ) where


import           Control.Monad (when)


-- | Gradient descent configuration
data Config net = Config
  { maxIterNum :: Int
    -- ^ Number of iteration
  , scaleCoef :: Double
    -- ^ Gradient scaling coefficient
  , gradient :: net -> net
    -- ^ Net gradient on the training data
  , substract :: net -> net -> Double -> net
    -- ^ Multiple the second net by the given scaling factor and
    -- substract it from the first net
  , quality :: net -> Double
    -- ^ Net quality measure
  , reportEvery :: Int
    -- ^ How often report the quality
  }


-- | Perform simple gradient descent.
gradDesc :: net -> Config net -> IO net
gradDesc net0 Config{..} =
  go 0 net0
  where
    go iterNum net
      | iterNum <= maxIterNum = do
          when (iterNum `mod` reportEvery == 0) $ do
            print $ quality net
          let grad = gradient net
              newNet = substract net grad scaleCoef
          go (iterNum+1) newNet
      | otherwise = return net
