{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE TypeOperators #-}


module SMT
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

import           Basic
import qualified FeedForward as FFN
import           FeedForward (FFN(..))
import qualified GradientDescent as GD
import qualified Encoder as Enc
import           Encoder (Encoder)
import qualified Decoder as Dec
import           Decoder (Decoder)


----------------------------------------------
-- SMT
----------------------------------------------


-- | SMT
data SMT = SMT
  { _encoder :: Encoder
      5  -- input vector
      5  -- internal hidden state
      5  -- output hidden state
  , _decoder :: Decoder
      5  -- input hidden state (coming from encoder)
      5  -- internal hidden state 1
      5  -- output vector
      5  -- internal hidden state 2
  , _h0  :: R 5
    -- ^ The initial hidden state of the encoder
  }
  deriving (Generic)

instance BP.Backprop SMT

makeLenses ''SMT


-- | New random Encoder-Decoder SMT system.
new 
  :: R 5 -- ^ EOS
  -> IO SMT
new eos =
  SMT <$> Enc.new 5 5 5 <*> Dec.new 5 5 5 5 eos <*> vector 5
  -- SMT <$> Enc.new 5 3 1 <*> Dec.new 1 1 5 1 eos <*> vector 3? 5?


run
  :: Reifies s W
  => BVar s SMT     -- ^ SMT network
  -> [BVar s (R 5)] -- ^ Input sentence
  -> [BVar s (R 5)] -- ^ Output sentence
  -> BVar s Double  -- ^ log P(output | input)
run smt inp out =
  -- encode the input; by applying `head` we only take the last hidden state of
  -- the encoder
  let inputEnc = head $ Enc.run (smt ^^. encoder) inp
  in  Dec.run (smt ^^. decoder) inputEnc out


-- | Substract the second network from the first one.
substract :: SMT -> SMT -> Double -> SMT
substract x y coef = SMT
  { _encoder = Enc.substract (_encoder x) (_encoder y) coef
  , _decoder = Dec.substract (_decoder x) (_decoder y) coef
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
type TrainElem = ([R 5], [R 5])


-- | Training dataset (both good and bad examples)
data Train = Train
  { goodSet :: [TrainElem]
  , badSet :: [TrainElem]
  }


-- -- | Log-likelihood of the training dataset
-- logLL
--   :: Reifies s W
--   => [TrainElem]
--   -> BVar s RNN
--   -> BVar s Double
-- logLL dataSet net
--   = sum
--   . map (runRNN net . map BP.auto)
--   $ dataSet


-- | Normalized log-likelihood of the training dataset
logLL
  :: Reifies s W
  => [TrainElem]
  -> BVar s SMT
  -> BVar s Double
logLL dataSet net =
  sum
    [ run net (map BP.auto inp) (map BP.auto out)
    | (inp, out) <- dataSet 
    ]


-- -- | Normalized log-likelihood of the training dataset
-- normLogLL
--   :: Reifies s W
--   => [TrainElem]
--   -> BVar s SMT
--   -> BVar s Double
-- normLogLL dataSet net =
--   sum
--     [ logProb / n
--     | (inp, out) <- dataSet
--     , let logProb = 
--             run net (map BP.auto inp) (map BP.auto out)
--     , let n = fromIntegral $ length out + 1
--     ]


-- | Quality of the network (inverted; the lower the better)
qualityInv
  :: Reifies s W
  => Train
  -> BVar s SMT
  -> BVar s Double
qualityInv Train{..} net =
  negLLL goodSet net - log (1 + negLLL badSet net)
  -- negLLL goodSet net - logBase 1.5 (1 + negLLL badSet net)
  where
    -- negLLL dataSet net = negate (normLogLL dataSet net)
    negLLL dataSet net = negate (logLL dataSet net)


----------------------------------------------
-- Gradient
----------------------------------------------


-- | Gradient calculation
calcGrad dataSet net =
  BP.gradBP (qualityInv dataSet) net


----------------------------------------------
-- Data
----------------------------------------------


-- | Word representations
one, two, three, four, eos :: R 5
one   = LA.vector [1, 0, 0, 0, 0]
two   = LA.vector [0, 1, 0, 0, 0]
three = LA.vector [0, 0, 1, 0, 0]
four  = LA.vector [0, 0, 0, 1, 0]
eos   = LA.vector [0, 0, 0, 0, 1]


goodData :: [TrainElem]
goodData =
--   [
--     ([], [])
--   , ([one], [one])
--   , ([two], [two])
--   , ([three], [three])
--   , ([four], [one, one])
--   ]
  [
    ([], [])
  , ([one], [one])
  , ([two], [one, one])
  , ([three], [one, one, one])
  , ([four], [one, one, one, one])
  ]


badData :: [TrainElem]
badData =
  [
--     ([one], [two])
  ]


-- | Training dataset (both good and bad examples)
trainData = Train
  { goodSet = goodData
  , badSet = badData 
  }


----------------------------------------------
-- Main
----------------------------------------------


main :: IO ()
main = do
  smt <- new eos
  smt' <- GD.gradDesc smt $ GD.Config
    { iterNum = 30000
    , scaleCoef = 0.05
    , gradient = calcGrad trainData
    , substract = substract
    , quality = BP.evalBP (qualityInv trainData)
    , reportEvery = 500 
    }
  -- print $ Dec._eos (_decoder smt')
  let test inp out = do
        let res = BP.evalBP0 $
              run (BP.auto smt') (map BP.auto inp) (map BP.auto out)
        putStrLn $ show res ++ " (out length: " ++ show (length out) ++ ")"
--   putStrLn "# good:"
--   test [] []
--   test [one] [one]
--   test [two] [two]
--   test [three] [three]
--   test [four] [one, one]
--   putStrLn "# bad:"
--   test [two] []
--   test [one] [two]
--   test [two] [three]
--   test [four] []
--   test [four] [one]
--   test [four] [one, one, one]
  putStrLn "# good:"
  test [] []
  test [one] [one]
  test [two] [one, one]
  test [three] [one, one, one]
  test [four] [one, one, one, one]
  putStrLn "# bad:"
  test [two] []
  test [one] [two]
  test [two] [three]
  test [four] []
  test [four] [one]
  test [four] [two]
