{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}


module Ex4 where


import           Prelude hiding (words)

import           GHC.Generics (Generic)

import           Lens.Micro.TH (makeLenses)

import qualified Numeric.LinearAlgebra.Static as LA
import qualified Numeric.LinearAlgebra        as LAD

import qualified Numeric.Backprop as BP
import           Numeric.Backprop ((^^.))
import           Numeric.LinearAlgebra.Static.Backprop
                   (R, L, (#), (#>), dot)

import           Basic
import qualified GradientDescent as GD
import qualified FeedForward as FFN
import           FeedForward (FFN(..))


----------------------------------------------
-- RNN
----------------------------------------------


-- Recursive Neural Network
data RNN = RNN
  { _ffG :: FFN
      3  -- RNN's hidden state
      3  -- ffG's internal hidden state
      3  -- ffG's output size (size of the vocabulary, including EOS)
  , _ffB :: FFN
      6  -- ffB takes on input the current word + the previous RNN's hidden state
      3  -- ffB's internal hidden state
      3  -- ffB's output size (the next RNN's hidden state)
  , _h0  :: R 3
    -- ^ The initial RNN's hidden state
  }
  deriving (Show, Generic)

instance BP.Backprop RNN
makeLenses ''RNN


-- run the netwok: determine the probability
-- of the given list of words
runRNN net words =
  go (net ^^. h0) (log 1.0) words
  where
    -- run the calculation, given the previous hidden state
    -- and the list of words to generate
    go hPrev prob (wordVect : ws) =
      let
        -- determine the probability vector
        probVect = softmax $ FFN.run (net ^^. ffG) hPrev
        -- determine the actual probability of the current word
        newProb = log $ probVect `dot` wordVect
        -- determine the next hidden state
        hNext = FFN.run (net ^^. ffB) (wordVect # hPrev)
      in
        go hNext (prob + newProb) ws
    -- if the list of words is empty
    go hPrev prob [] =
      let
        -- determine the probability vector
        probVect = softmax $ FFN.run (net ^^. ffG) hPrev
        -- determine the actual probability of EOS
        newProb = log $ probVect `dot` BP.constVar eos
      in
        prob + newProb


-- evaluate the network (user-friendly version
-- of `runRNN`)
evalRNN net input =
  BP.evalBP0 
    ( runRNN
        (BP.constVar net)
        (map BP.constVar input)
    )


----------------------------------------------
-- Likelihood
----------------------------------------------


-- | Log-likelihood of the training dataset
logLL dataSet net
  = sum
  . map (runRNN net . map BP.constVar)
  $ dataSet


-- | Quality of the network (inverted; the lower the better)
invQuality dataSet net =
  negate (logLL dataSet net)


----------------------------------------------
-- Training
----------------------------------------------


-- vocabulary, including EOS
one, two, eos :: R 3
one   = LA.vector [1, 0, 0]
two   = LA.vector [0, 1, 0]
eos   = LA.vector [0, 0, 1]


-- training data (sequences of alternating
-- one's and two's)
trainData=
  [ [one]
  , [one, two]
  , [one, two, one]
  , [one, two, one, two]
--   , [one, two, one, two, one]
--   , [one, two, one, two, one, two]
--   , [one, two, one, two, one, two, one]
--   , [one, two, one, two, one, two, one, two]
  ]


-- train with the default dataset
train net =
  trainWith trainData net


-- train with a custom dataset
trainWith dataSet net =
  GD.gradDesc net (gdCfg dataSet)


-- main function
main = do
  net <- newRNN
  train net


-----------------------------------------------------
-- Gradient descent
-----------------------------------------------------


-- gradient descent configuration
gdCfg dataSet = GD.Config
  { iterNum = 10000
  , scaleCoef = 0.01
  , gradient = BP.gradBP (invQuality dataSet)
  , substract = subRNN
  , quality = BP.evalBP (invQuality dataSet)
  , reportEvery = 500
  }


-- substract the second network from the first one
subRNN x y coef = RNN
  { _ffG = FFN.substract (_ffG x) (_ffG y) coef
  , _ffB = FFN.substract (_ffB x) (_ffB y) coef
  , _h0 = _h0 x - scale coef (_h0 y)
  }


-----------------------------------------------------
-- Random data construction
-----------------------------------------------------


-- create a new, random RRN
newRNN = do
  ffg <- FFN.new 3 3 3
  ffb <- FFN.new 6 3 3
  rnn <- RNN ffg ffb <$> vector 3
  return rnn


-----------------------------------------------------
-- Backup
-----------------------------------------------------


-- | Normalized log-likelihood of the training dataset
normLogLL dataSet net =
  sum
    [ logProb / n
    | trainElem <- dataSet
    , let logProb = runRNN net (map BP.constVar trainElem)
    , let n = fromIntegral $ length trainElem + 1
    ]
