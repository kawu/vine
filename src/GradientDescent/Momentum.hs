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

--   , scaleCoef :: Double
--     -- ^ Gradient scaling coefficient

  , gradient :: net -> net
    -- ^ Net gradient on the training data
--   , substract :: net -> net -> Double -> net
--     -- ^ Multiple the second net by the given scaling factor and
--     -- substract it from the first net
  , quality :: net -> Double
    -- ^ Net quality measure
  , reportEvery :: Int
    -- ^ How often report the quality

  , gain0     :: Double
  -- ^ Initial gain parameter
  , tau       :: Double
  -- ^ After how many iterations over the entire dataset
  -- the gain parameter is halved
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

  -- | Size
  size :: p -> Double


-- | Perform simple gradient descent with momentum.
gradDesc :: (ParamSet net) => net -> Config net -> IO net
gradDesc net0 Config{..} =

  go 0 zero net0

  where

    -- Gain in the k-th iteration
    gain k = (gain0 * tau) / (tau + done k)

    -- Number of completed iterations over the full dataset.
    done :: Int -> Double
    done k = fromIntegral (iterNum - k)

    go k momentum net
      | k > iterNum = return net
      | otherwise = do
          when (k `mod` reportEvery == 0) $ do
            putStr . show $ quality net
            putStrLn $ " (size = " ++ show (size net) ++ ")"
          let grad = scale (gain k) (gradient net)
              momentum' = scale gamma momentum `add` grad
              newNet = net `sub` momentum'
          go (k+1) momentum' newNet
