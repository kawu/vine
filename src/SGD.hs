{-# LANGUAGE RecordWildCards #-}


module SGD
  ( DataSet (..)
  , Config(..)
  , sgd
  ) where


import           Control.Monad (forM_)

import qualified System.Random as R

import qualified Data.IORef as IO

import qualified GradientDescent.Momentum as Mom


----------------------------------------------
-- Types
----------------------------------------------


-- | Encoded dataset stored on a disk
data DataSet elem = DataSet
  { size :: Int 
    -- ^ The size of the dataset; the individual indices are
    -- [0, 1, ..., size - 1]
  , elemAt :: Int -> IO elem
    -- ^ Get the dataset element with the given identifier
  }


-- | SGD configuration
data Config net elem = Config
  { iterNum :: Int
    -- ^ Number of iteration over the entire training dataset
  
  , batchSize :: Int
    -- ^ Size of the SGD batch

  , gradient :: [elem] -> net -> net
    -- ^ Net gradient on the given dataset
  , quality :: elem -> net -> Double
    -- ^ Net quality measure w.r.t. the given dataset element
    --
    -- IMPORTANT: we assume here that the quality on a dataset is the sum of
    -- the qualities on its individual elements

  , reportEvery :: Double
    -- ^ How often report the quality (with `1` meaning once per pass over the
    -- training data)

  , gain0     :: Double
  -- ^ Initial gain parameter
  , tau       :: Double
  -- ^ After how many passes over the training data the gain parameter is
  -- halved
  , gamma     :: Double
  -- ^ The momentum-related parameter (TODO: explain)
  }


----------------------------------------------
-- SGD
----------------------------------------------


-- | Perform stochastic gradient descent with momentum.
sgd
  :: (Mom.ParamSet net)
  => net
  -> DataSet elem
  -> Config net elem
  -> IO net
sgd net0 dataSet Config{..} = do
  Mom.gradDesc net0 cfg
  where
    cfg = Mom.Config
      { Mom.iterNum = ceiling
          $ fromIntegral (size dataSet * iterNum)
          / fromIntegral batchSize
      , Mom.gradient = \net -> do
          sample <- randomSample batchSize dataSet
          return $ gradient sample net
      , Mom.quality = \net -> do
          res <- IO.newIORef 0.0
          forM_ [0 .. size dataSet - 1] $ \ix -> do
            elem <- elemAt dataSet ix
            IO.modifyIORef' res (+ quality elem net)
          IO.readIORef res
      -- TODO: we could repot on a random sample!
      -- That could be also done more often!
      , Mom.reportEvery = ceiling
          $ fromIntegral (size dataSet) * reportEvery
          / fromIntegral batchSize
      , Mom.gain0 = gain0
      , Mom.tau
          = fromIntegral (size dataSet) * tau
          / fromIntegral batchSize
      , Mom.gamma = gamma
      }


-- | Random dataset sample
randomSample :: Int -> DataSet a -> IO [a]
randomSample k dataSet
  | k <= 0 = return []
  | otherwise = do
      ix <- R.randomRIO (0, size dataSet - 1)
      x <- elemAt dataSet ix
      (x:) <$> randomSample (k-1) dataSet
