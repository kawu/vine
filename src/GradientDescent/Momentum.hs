{-# LANGUAGE RecordWildCards #-}


module GradientDescent.Momentum
  ( Config(..)
  , ParamSet(..)
  , gradDesc
  ) where


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
  , reportEvery :: Int
    -- ^ How often report the quality

  , gain0     :: Double
  -- ^ Initial gain parameter
  , tau       :: Double
  -- ^ After how many gradient calculations the gain parameter is halved
  , gamma     :: Double
  -- ^ The momentum-related parameter (TODO: explain)
  }


class ParamSet p where
  -- | Zero
  zero :: p
  -- | Scaling
  scale :: Double -> p -> p
  -- | Negation
  neg :: p -> p
  neg = scale (-1)
  -- | Addition
  add :: p -> p -> p
  add x y = x `sub` neg y
  -- | Substruction
  sub :: p -> p -> p
  sub x y = x `add` neg y
 
  -- | Net size
  size :: p -> Double


-- | Perform simple gradient descent with momentum.
gradDesc :: (ParamSet net) => net -> Config net -> IO net
gradDesc net0 Config{..} =

  go 0 zero net0

  where

    -- Gain in the k-th iteration
    gain k
      = (gain0 * tau)
      / (tau + fromIntegral k)

    go k momentum net
      | k > iterNum = return net
      | otherwise = do
          let netSize = size net
          when (k `mod` reportEvery == 0) $ do
            putStr . show =<< quality net
            putStrLn $ " (size = " ++ show netSize ++ ")"
          grad <- netSize `seq`
            (scale (gain k) <$> gradient net)
          let momentum' = scale gamma momentum `add` grad
              newNet = net `sub` momentum'
          go (k+1) momentum' newNet
