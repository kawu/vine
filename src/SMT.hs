{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE TypeOperators #-}


module SMT where


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

import           Net.Basic
import qualified Net.FeedForward as FFN
import           Net.FeedForward (FFN(..))
import qualified GradientDescent as GD
import qualified SMT.Encoder as Enc
import           SMT.Encoder (Encoder)
import qualified SMT.Decoder as Dec
import           SMT.Decoder (Decoder)


----------------------------------------------
-- SMT
----------------------------------------------


-- | Encoder-Decoder SMT system
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
newSMT eos =
  SMT <$> Enc.new 5 5 5 <*> Dec.new 5 5 5 5 eos <*> vector 5


-- | Calculate the log probability of the output
-- sentence `out` given the input sentence `inp`.
runSMT smt inp out =
  -- encode the input; by applying `head` we only
  -- take the last hidden state of the encoder
  let inputEnc = head $ Enc.run (smt ^^. encoder) inp
  in  Dec.run (smt ^^. decoder) inputEnc out



-- | User-friendly version of `run`
evalSMT smt inp out =
  BP.evalBP0 
    ( runSMT
        (BP.constVar smt)
        (map BP.constVar inp)
        (map BP.constVar out)
    )


----------------------------------------------
-- Likelihood
----------------------------------------------


-- | Log-likelihood of the training dataset
logLL dataSet net =
  sum
    [ runSMT net
        (map BP.constVar inp)
        (map BP.constVar out)
    | (inp, out) <- dataSet ]


-- | Quality of the network (inverted; the lower the better)
invQuality dataSet net =
  negate (logLL dataSet net)


----------------------------------------------
-- Gradient
----------------------------------------------


-- | Gradient calculation
calcGrad dataSet net =
  BP.gradBP (invQuality dataSet) net


----------------------------------------------
-- Training
----------------------------------------------


-- | Word representations
one, two, three, four, eos :: R 5
one   = LA.vector [1, 0, 0, 0, 0]
two   = LA.vector [0, 1, 0, 0, 0]
three = LA.vector [0, 0, 1, 0, 0]
four  = LA.vector [0, 0, 0, 1, 0]
eos   = LA.vector [0, 0, 0, 0, 1]


-- training data:
-- 1 -> 1
-- 2 -> 1, 1
-- 3 -> 1, 1, 1
-- 4 -> 1, 1, 1, 1
trainData =
  [
    ([], [])
  , ([one], [one])
  , ([two], [one, one])
  , ([three], [one, one, one])
  , ([four], [one, one, one, one])
  ]


-- another training dataset
trainData2 =
  [
    ([], [])
  , ([one], [one])
  , ([two], [two])
  , ([three], [three])
  , ([four], [one, one])
  ]


-- train with the default dataset
train net =
  trainWith trainData net


-- train with a custom dataset
trainWith dataSet net =
  GD.gradDesc net (gdCfg dataSet)


-----------------------------------------------------
-- Gradient descent
-----------------------------------------------------


-- gradient descent configuration
gdCfg dataSet = GD.Config
  { iterNum = 10000
  , scaleCoef = 0.025
  , gradient = calcGrad dataSet
  , substract = substract
  , quality = BP.evalBP (invQuality dataSet)
  , reportEvery = 500 
  }


-- | Substract the second network from the first one.
substract :: SMT -> SMT -> Double -> SMT
substract x y coef = SMT
  { _encoder = Enc.substract (_encoder x) (_encoder y) coef
  , _decoder = Dec.substract (_decoder x) (_decoder y) coef
  , _h0 = _h0 x - scale coef (_h0 y)
  }


----------------------------------------------
-- Backup
----------------------------------------------


-- main :: IO ()
-- main = do
--   smt <- new eos
--   smt' <- GD.gradDesc smt (gdCfg trainData)
--   let test inp out = do
--         let res = BP.evalBP0 $
--               run (BP.constVar smt') (map BP.constVar inp) (map BP.constVar out)
--         putStrLn $ show res ++ " (out length: " ++ show (length out) ++ ")"
-- --   putStrLn "# good:"
-- --   test [] []
-- --   test [one] [one]
-- --   test [two] [two]
-- --   test [three] [three]
-- --   test [four] [one, one]
-- --   putStrLn "# bad:"
-- --   test [two] []
-- --   test [one] [two]
-- --   test [two] [three]
-- --   test [four] []
-- --   test [four] [one]
-- --   test [four] [one, one, one]
--   putStrLn "# good:"
--   test [] []
--   test [one] [one]
--   test [two] [one, one]
--   test [three] [one, one, one]
--   test [four] [one, one, one, one]
--   putStrLn "# bad:"
--   test [two] []
--   test [one] [two]
--   test [two] [three]
--   test [four] []
--   test [four] [one]
--   test [four] [two]
