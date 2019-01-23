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
one, two, three, end, eos :: R 5
one   = LA.vector [1, 0, 0, 0, 0]
two   = LA.vector [0, 1, 0, 0, 0]
three = LA.vector [0, 0, 1, 0, 0]
end   = LA.vector [0, 0, 0, 1, 0]
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
    -- ^ Probability of the sentence
runRNN net =
  go (net ^^. h0) 1.0
  where
    -- run the calculation, given the previous hidden state
    -- and the list of words to generate
    go hPrev prob (wordVect : ws) =
      let
        -- determine the probability vector
        probVect = softmax $ runFFN (net ^^. ffG) hPrev
        -- determine the actual probability of the current word
        newProb = probVect `LBP.dot` wordVect
        -- determine the next hidden state
        hNext = runFFN (net ^^. ffB) (wordVect # hPrev)
      in
        go hNext (prob * newProb) ws
    go hPrev prob [] =
      let
        -- determine the probability vector
        probVect = softmax $ runFFN (net ^^. ffG) hPrev
        -- determine the actual probability of EOS
        newProb = probVect `LBP.dot` BP.auto eos
      in
        newProb


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
-- Error
----------------------------------------------


type TrainElem = [R 5]
type Train = [TrainElem]


-- | Likelihood of the training dataset
likelihood
  :: Reifies s W
  => Train
  -> BVar s RNN
  -> BVar s Double
likelihood dataSet net
  = product
  . map (runRNN net . map BP.auto)
  $ dataSet


-- | Log-likelihood
logLL
  :: Reifies s W
  => Train
  -> BVar s RNN
  -> BVar s Double
logLL dataSet =
  (\x->(-x)) . log . likelihood dataSet


----------------------------------------------
-- Gradient Descent
----------------------------------------------


calcGrad dataSet net =
  BP.gradBP (logLL dataSet) net


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
      print $ BP.evalBP (logLL dataSet) net
      let grad = calcGrad dataSet net
          newNet = subRNN net grad coef
      gradDesc (iterNum-1) coef dataSet newNet
  | otherwise = return net


----------------------------------------------
-- Main
----------------------------------------------


-- | The dataset consists of pairs (input, target output)
trainData :: Train
trainData =
  [ [end]
  , [one, two, three, end]
  -- , [one, two, three, four, one, two, three, four, end]
  ]


main :: IO ()
main = do
  -- an actual network
  let list k = take k $ cycle [0, 1, 2]
      zero k = take k $ repeat 0
      ffg = FFN
        { _nWeights1 = LA.matrix $ zero (5*5)
        , _nBias1 = LA.vector $ zero 5
        , _nWeights2 = LA.matrix $ zero (5*5)
        , _nBias2 = LA.vector $ zero 5
        } :: FFN 5 5 5
      ffb = FFN
        { _nWeights1 = LA.matrix $ zero (5*10)
        , _nBias1 = LA.vector $ zero 5
        , _nWeights2 = LA.matrix $ zero (5*5)
        , _nBias2 = LA.vector $ zero 5
        } :: FFN 10 5 5
      rnn = RNN
        { _ffG = ffg
        , _ffB = ffb
        , _h0 = LA.vector $ zero 5
        } :: RNN
  rnn' <- gradDesc 1000 0.01 trainData rnn
  let res1 = runRNN (BP.auto rnn')
               (map BP.auto [end])
      res2 = runRNN (BP.auto rnn')
               (map BP.auto [one, two, three, end])
      res3 = runRNN (BP.auto rnn')
               (map BP.auto [three, two, one, end, three])
  print $ BP.evalBP0 res1
  print $ BP.evalBP0 res2
  print $ BP.evalBP0 res3
--       resZers = head $ runRNN (BP.auto rnn')
--               (map BP.auto [zers, zers, zers, zers, zers])
--       resMix1 = head $ runRNN (BP.auto rnn')
--               (map BP.auto [ones, zers, zers, zers, zers, zers, zers, zers])
--       resMix2 = head $ runRNN (BP.auto rnn')
--               (map BP.auto [zers, zers, ones, zers, zers])
--       resMix3 = head $ runRNN (BP.auto rnn')
--               (map BP.auto [zers, zers, zers, zers, zers, zers, zers, ones])
--       resMix4 = head $ runRNN (BP.auto rnn')
--               (map BP.auto [zers, zers, zers, zers, ones, zers, zers, zers])
--   LBP.disp 5 $ BP.evalBP0 resMix1
--   LBP.disp 5 $ BP.evalBP0 resMix2
--   LBP.disp 5 $ BP.evalBP0 resMix3
--   LBP.disp 5 $ BP.evalBP0 resMix4
  return ()


----------------------------------------------
-- Rendering
----------------------------------------------


-- showVect :: (KnownNat n) => R n -> String
-- showVect = undefined
