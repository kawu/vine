{-# LANGUAGE RecordWildCards #-}


module GradientDescent.Adam
  ( Config(..)
  , ParamSet(..)
  , adam
  ) where


import           Prelude hiding (div)

import           Control.Monad (when)


-- | Gradient descent configuration
data Config net = Config
  { iterNum :: Int
    -- ^ Number of iteration

  , gradient :: net -> IO net
    -- ^ Net gradient on the training data; embedded in the IO monad because
    -- it's stochastic (not a pure function)
  , quality :: net -> IO Double
    -- ^ Net quality measure
  , size :: net -> Double
    -- ^ Net size
  , reportEvery :: Int
    -- ^ How often report the quality

  , alpha :: Double
    -- ^ Step size
  , beta1 :: Double
    -- ^ 1st exponential moment decay
  , beta2 :: Double
    -- ^ 1st exponential moment decay
  , eps   :: Double
    -- ^ Epsilon
  }


class ParamSet p where
  -- | Zero
  zero :: p
  -- | Mapping
  pmap :: (Double -> Double) -> p -> p

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


-- | Perform simple gradient descent with momentum.
adam :: (ParamSet net) => net -> Config net -> IO net
adam net0 Config{..} =

  go 1 zero zero net0

  where

--     -- Gain in the k-th iteration
--     gain t = (gain0 * tau) / (tau + done t)

--     -- Number of completed iterations over the full dataset.
--     done :: Int -> Double
--     done t = fromIntegral (iterNum - t)

    -- `m` is the 1st moment, `v` is the 2nd moment
    go t m v net
      | t > iterNum = return net
      | otherwise = do
          let netSize = size net
          when (t `mod` reportEvery == 0) $ do
            putStr . show =<< quality net
            putStrLn $ " (size = " ++ show netSize ++ ")"
          g <- netSize `seq` gradient net
          let m' = pmap (*beta1) m `add` pmap (*(1-beta1)) g
              v' = pmap (*beta2) v `add` pmap (*(1-beta2)) (g `mul` g)
              -- bias-corrected moment estimates 
              mb = pmap (/(1-beta1^t)) m'
              vb = pmap (/(1-beta2^t)) v'
              newNet = net `sub`
                ( pmap (*alpha) mb `div`
                  (pmap (+eps) (pmap sqrt vb))
                )
          go (t+1) m' v' newNet
