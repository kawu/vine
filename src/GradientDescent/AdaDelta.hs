{-# LANGUAGE RecordWildCards #-}


module GradientDescent.AdaDelta
  ( Config(..)
  , ParamSet(..)
  , gradDesc
  ) where


import           Prelude hiding (div)
import           Control.Monad (when)


-- | Gradient descent configuration
data Config net = Config
  { iterNum :: Int
    -- ^ Number of iteration

  , gradient :: net -> IO net
    -- ^ Net gradient on the training data; embedded in the IO monad so that
    -- it can randomly choose a subset of the training data
  , quality :: net -> IO Double
    -- ^ Net quality measure
  , size :: net -> Double
    -- ^ Net size
  , reportEvery :: Int
    -- ^ How often report the quality

  , gamma :: Double
    -- ^ Decay parameter
  , eps   :: Double
    -- ^ Epsilon value
  }


class ParamSet p where
  -- | Zero
  zero :: p
  -- | Mapping
  pmap :: (Double -> Double) -> p -> p
  -- | Scaling
  scale :: Double -> p -> p
  scale x = pmap (*x)

  -- | Negation
  neg :: p -> p
  neg = pmap (\x -> -x)
  -- | Addition
  add :: p -> p -> p
  add x y = x `sub` neg y
  -- | Substruction
  sub :: p -> p -> p
  sub x y = x `add` neg y

  -- | Element-wise multiplication
  mul :: p -> p -> p
  mul x y = x `div` pmap (1.0/) y
  -- | Element-wise division
  div :: p -> p -> p
  div x y = x `mul` pmap (1.0/) y

  -- | Root square
  squareRoot :: p -> p
  squareRoot = pmap sqrt

  -- | Square
  square :: p -> p
  square x = x `mul` x


-- | Perform simple gradient descent with momentum.
gradDesc :: (ParamSet net) => net -> Config net -> IO net
gradDesc net0 Config{..} =

  go 0 zero zero zero net0

  where

    go k expSqGradPrev expSqDeltaPrev deltaPrev net
      | k > iterNum = return net
      | otherwise = do
          let netSize = size net
          when (k `mod` reportEvery == 0) $ do
            putStr . show =<< quality net
            putStrLn $ " (size = " ++ show netSize ++ ")"
          grad <- netSize `seq` gradient net
          let expSqGrad = scale gamma expSqGradPrev
                    `add` scale (1-gamma) (square grad)
              rmsGrad = squareRoot (pmap (+eps) expSqGrad)
              expSqDelta = scale gamma expSqDeltaPrev
                     `add` scale (1-gamma) (square deltaPrev)
              rmsDelta = squareRoot (pmap (+eps) expSqDelta)
              delta = (rmsDelta `mul` grad) `div` rmsGrad
              -- delta = scale 0.01 grad `div` rmsGrad
              newNet = net `sub` delta
          go (k+1) expSqGrad expSqDelta delta newNet
