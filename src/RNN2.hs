{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE TypeOperators #-}


{- 
    Simple language modeling RNN
-}



module RNN2
  ( main
  ) where


import           Prelude hiding (words)

import           GHC.Generics (Generic)
import           GHC.TypeNats (KnownNat)
import qualified GHC.TypeNats as Nats

import           Control.Monad (forM, forM_)

import           System.Random (randomRIO)

-- import           Control.Lens.TH (makeLenses)
import           Lens.Micro.TH (makeLenses)
import           Lens.Micro ((^.))

import           Data.Maybe (fromJust)
import qualified Numeric.Backprop as BP
import           Numeric.Backprop ((^^.))
import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
import           Numeric.LinearAlgebra.Static.Backprop
  (R, L, BVar, Reifies, W, (#))
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


-- | Apply the softmax layer to a vector
softmax
  :: (KnownNat n, Reifies s W)
  => BVar s (R n)
  -> BVar s (R n)
softmax x0 =
  LBP.vmap (/norm) x
  where
    x = LBP.vmap' exp x0
    norm = LBP.norm_1V x


-- -- myFunc :: Double -> Double
-- myFunc :: Floating a => a -> a
-- myFunc x = sqrt (x * 4)


----------------------------------------------
-- Feed-forward net
----------------------------------------------


data FFN idim hdim odim = FFN
  { _nWeights1 :: L hdim idim
  , _nBias1    :: R hdim
  , _nWeights2 :: L odim hdim
  , _nBias2    :: R odim
  }
  deriving (Show, Generic)

instance (KnownNat idim, KnownNat hdim, KnownNat odim) 
  => BP.Backprop (FFN idim hdim odim)

makeLenses ''FFN


runFFN
  :: (KnownNat idim, KnownNat hdim, KnownNat odim, Reifies s W)
  => BVar s (FFN idim hdim odim)
  -> BVar s (R idim)
  -> BVar s (R odim)
runFFN net x = z
  where
    -- run first layer
    y = logistic $ (net ^^. nWeights1) #> x + (net ^^. nBias1)
    -- run second layer
    z = relu $ (net ^^. nWeights2) #> y + (net ^^. nBias2)
    -- z = (net ^^. nWeights2) #> y + (net ^^. nBias2)


-- | Substract the second network from the first one.
subFFN 
  :: (KnownNat idim, KnownNat hdim, KnownNat odim)
  => FFN idim hdim odim
  -> FFN idim hdim odim
  -> Double 
  -> FFN idim hdim odim
subFFN x y coef = FFN
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


----------------------------------------------
-- Words
----------------------------------------------


-- | The EOS vector
one, two, three, four, eos :: R 5
one   = LA.vector [1, 0, 0, 0, 0]
two   = LA.vector [0, 1, 0, 0, 0]
three = LA.vector [0, 0, 1, 0, 0]
four  = LA.vector [0, 0, 0, 1, 0]
eos   = LA.vector [0, 0, 0, 0, 1]


----------------------------------------------
-- RNN
----------------------------------------------


-- Recursive Neural Network
data RNN = RNN
  { _ffG :: FFN
      5  -- RNN's hidden state
      5  -- ffG's internal hidden state
      5  -- ffG's output size (size of the vocabulary, including EOS)
  , _ffB :: FFN
      10 -- ffB takes on input the current word + the previous RNN's hidden state
      5  -- ffB's internal hidden state
      5  -- ffB's output size (the next RNN's hidden state)
  , _h0  :: R 5
    -- ^ The initial RNN's hidden state
  }
  deriving (Show, Generic)

instance BP.Backprop RNN

makeLenses ''RNN


runRNN
  :: (Reifies s W)
  => BVar s RNN
  -> [BVar s (R 5)]
    -- ^ Sentence (sequence of vector representations)
  -> BVar s Double
    -- ^ Probability of the sentence (log-domain!)
runRNN net =
  go (net ^^. h0) (log 1.0)
  where
    -- run the calculation, given the previous hidden state
    -- and the list of words to generate
    go hPrev prob (wordVect : ws) =
      let
        -- determine the probability vector
        probVect = softmax $ runFFN (net ^^. ffG) hPrev
        -- determine the actual probability of the current word
        newProb = log $ probVect `LBP.dot` wordVect
        -- determine the next hidden state
        hNext = runFFN (net ^^. ffB) (wordVect # hPrev)
      in
        go hNext (prob + newProb) ws
    go hPrev prob [] =
      let
        -- determine the probability vector
        probVect = softmax $ runFFN (net ^^. ffG) hPrev
        -- determine the actual probability of EOS
        newProb = log $ probVect `LBP.dot` BP.auto eos
      in
        prob + newProb


-- | Substract the second network from the first one.
-- subRNN :: RNN -> RNN -> Double -> RNN
subRNN x y coef = RNN
  { _ffG = subFFN (_ffG x) (_ffG y) coef
  , _ffB = subFFN (_ffB x) (_ffB y) coef
  , _h0 = _h0 x - scale (_h0 y)
  }
  where
    scale x
      = fromJust
      . LA.create
      . LAD.scale coef
      $ LA.unwrap x


----------------------------------------------
-- Likelihood
----------------------------------------------


-- | Training dataset element
type TrainElem = [R 5]


-- | Training dataset (both good and bad examples)
data Train = Train
  { goodSet :: [TrainElem]
  , badSet :: [TrainElem]
  }


-- | Log-likelihood of the training dataset
logLL
  :: Reifies s W
  => [TrainElem]
  -> BVar s RNN
  -> BVar s Double
logLL dataSet net
  = sum
  . map (runRNN net . map BP.auto)
  $ dataSet


-- | Negated log-likelihood
negLogLL
  :: Reifies s W
  => [TrainElem]
  -> BVar s RNN
  -> BVar s Double
negLogLL dataSet net = negate $ logLL dataSet net


-- | Quality of the network.  The lower the better...
qualityInv
  :: Reifies s W
  => Train
  -> BVar s RNN
  -> BVar s Double
qualityInv Train{..} net =
  negLogLL goodSet net - log (1 + negLogLL badSet net)


----------------------------------------------
-- Gradient Descent
----------------------------------------------


calcGrad dataSet net =
  BP.gradBP (qualityInv dataSet) net


gradDesc
  :: Int
  -- ^ Number of iterations
  -> Double
  -- ^ Gradient scaling coefficient
  -> Train
  -- ^ Training dataset
  -> RNN
  -- ^ Initial network
  -> IO RNN
  -- ^ Resulting network
gradDesc iterNum coef dataSet net
  | iterNum > 0 = do
      print $ BP.evalBP (qualityInv dataSet) net
      let grad = calcGrad dataSet net
          newNet = subRNN net grad coef
      gradDesc (iterNum-1) coef dataSet newNet
  | otherwise = return net


----------------------------------------------
-- Main
----------------------------------------------


goodData :: [TrainElem]
goodData =
  [ [one]
  , [one, two]
  , [one, two, one]
  , [one, two, one, two]
  -- additional
  , [one, two, one, two, one]
  , [one, two, one, two, one, two]
  , [one, two, one, two, one, two, one]
  , [one, two, one, two, one, two, one, two]
  ]


badData :: [TrainElem]
badData = 
  [ [two]
  , [one, one]
  , [three]
  , [four]
  , [eos]
  -- additional
  , [one, three]
  , [one, four]
  , [one, eos]
  , [one, two, three]
  , [one, two, four]
  , [one, two, eos]
  ]


-- | Training dataset (both good and bad examples)
trainData = Train
  { goodSet = goodData
  , badSet = badData 
  }


-- | A random list of values between 0 and 1
randomList :: Int -> IO [Double]
randomList 0 = return []
randomList n = do
  r  <- randomRIO (0, 1)
  rs <- randomList (n-1)
  return (r:rs) 


-- | Create a random matrix
matrix
  :: (KnownNat n, KnownNat m)
  => Int -> Int -> IO (L m n)
matrix n m = do
  list <- randomList (n*m)
  return $ LA.matrix list


-- | Create a random vector
vector :: (KnownNat n) => Int -> IO (R n)
vector k = do
  list <- randomList k
  return $ LA.vector list


main :: IO ()
main = do
  ffg <- FFN <$> matrix 5 5 <*> vector 5 <*> matrix 5 5 <*> vector 5
  ffb <- FFN <$> matrix 5 10 <*> vector 5 <*> matrix 5 5 <*> vector 5
  rnn <- RNN ffg ffb <$> vector 5
  rnn' <- gradDesc 20000 0.01 trainData rnn
  let test input =
        print $ BP.evalBP0 $
          runRNN (BP.auto rnn') (map BP.auto input)
  putStrLn "# good:"
  test [one]
  test [one, two]
  test [one, two, one]
  test [one, two, one, two]
  test [one, two, one, two, one, two, one, two]
  test [one, two, one, two, one, two, one, two, one, two]
  putStrLn "# bad:"
  test [two]
  test [one, one]
  test [two, two, two]
  test []


----------------------------------------------
-- Rendering
----------------------------------------------


-- showVect :: (KnownNat n) => R n -> String
-- showVect = undefined
