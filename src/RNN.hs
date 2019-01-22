{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}


module RNN
  ( main
  ) where


import           GHC.Generics (Generic)
-- import           GHC.TypeNats (KnownNat)

import           Control.Monad (forM, forM_)

import           System.Random (randomRIO)

-- import           Control.Lens.TH (makeLenses)
import           Lens.Micro.TH (makeLenses)
import           Lens.Micro ((^.))

import           Data.Maybe (fromJust)
import qualified Numeric.Backprop as BP
import           Numeric.Backprop ((^^.))
import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
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


-- -- myFunc :: Double -> Double
-- myFunc :: Floating a => a -> a
-- myFunc x = sqrt (x * 4)


----------------------------------------------
-- Configuration
----------------------------------------------


-- -- | Size of input vectors
-- inpDim :: Int
-- inpDim = 5
-- 
-- 
-- -- | Size of hidden vectors
-- hidDim :: Int
-- hidDim = 5
-- 
-- 
-- -- | Output layer size
-- outDim :: Int
-- outDim = 5


----------------------------------------------
-- Feed-forward net
----------------------------------------------


data FFN = FFN
  { _nWeights1 :: LBP.L 5 10
  , _nBias1    :: LBP.R 5
  , _nWeights2 :: LBP.L 5 5
  , _nBias2    :: LBP.R 5
  }
  deriving (Show, Generic)

instance BP.Backprop FFN

makeLenses ''FFN


runFFN net x = z
  where
    -- run first layer
    y = logistic $ (net ^^. nWeights1) #> x + (net ^^. nBias1)
    -- run second layer
    z = relu $ (net ^^. nWeights2) #> y + (net ^^. nBias2)
    -- z = (net ^^. nWeights2) #> y + (net ^^. nBias2)


-- | Substract the second network from the first one.
subFFN :: FFN -> FFN -> Double -> FFN
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
-- RNN
----------------------------------------------


-- Recursive Neural Network
data RNN = RNN
  { _ffn :: FFN
    -- ^ The underlying network used to calculate subsequent hidden values
  , _h0  :: LBP.R 5
    -- ^ The initial hidden state
  }
  deriving (Show, Generic)

instance BP.Backprop RNN

makeLenses ''RNN


runRNN
  :: LBP.Reifies s LBP.W
  => LBP.BVar s RNN
  -> [LBP.BVar s (LA.R 5)]
  -> [LBP.BVar s (LA.R 5)]
runRNN net [] = [net ^^. h0]
runRNN net (x:xs) = h:hs
  where
    -- run the recursive calculation (unless `null xs`)
    hs = runRNN net xs
    -- get the last hidden state
    h' = head hs
    -- calculate the resulting hidden value
    h = runFFN (net ^^. ffn) (x LBP.# h')


-- | Substract the second network from the first one.
subRNN :: RNN -> RNN -> Double -> RNN
subRNN x y coef = RNN
  { _ffn = subFFN (_ffn x) (_ffn y) coef
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


squaredError1
  :: LBP.Reifies s LBP.W
  => LBP.BVar s (LA.R 5)
  -> LBP.BVar s (LA.R 5)
  -> LBP.BVar s LA.ℝ
squaredError1 target output = err `LBP.dot` err
  where
    err = target - output


squaredError
  :: LBP.Reifies s LBP.W
  => [LBP.BVar s (LA.R 5)]
  -> [LBP.BVar s (LA.R 5)]
  -> LBP.BVar s LA.ℝ
squaredError targets outputs =
  go targets outputs -- (BP.sequenceVar outputs)
  where
    go ts os = 
      case (ts, os) of
        (t:tr, o:or) -> squaredError1 t o + go tr or
        ([], []) -> 0
        _ -> error "squaredError: lists of different size" 


-- netError1 target input net =
--   squaredError1
--     (BP.auto target)
--     (head $ runRNN net (map BP.auto input))


netError1 target input net =
  squaredError
    [BP.auto target]
    [head . runRNN net . map BP.auto $ input]


netError
  :: LBP.Reifies s LBP.W
  => Train
  -> LBP.BVar s RNN
  -> LBP.BVar s LA.ℝ
netError dataSet net =
  let
    inputs = map fst dataSet
    outputs = map (head . runRNN net . map BP.auto) inputs
    targets = map (BP.auto . snd) dataSet
  in  
    squaredError targets outputs


----------------------------------------------
-- Gradient Descent
----------------------------------------------


calcGrad1 target input net =
  BP.gradBP (netError1 target input) net


gradDesc1
  :: Int
  -- ^ Number of iterations
  -> [LA.R 5]
  -- ^ Input vectors
  -> LA.R 5
  -- ^ Target output vector
  -> RNN
  -- ^ Initial network
  -> IO RNN
  -- ^ Resulting network
gradDesc1 iterNum input target net
  | iterNum > 0 = do
      print $ BP.evalBP (netError1 target input) net
      let grad = calcGrad1 target input net
          newNet = subRNN net grad 0.01
      gradDesc1 (iterNum-1) input target newNet
  | otherwise = return net


calcGrad dataSet net =
  BP.gradBP (netError dataSet) net


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
      print $ BP.evalBP (netError dataSet) net
      let grad = calcGrad dataSet net
          newNet = subRNN net grad coef
      gradDesc (iterNum-1) coef dataSet newNet
  | otherwise = return net


----------------------------------------------
-- Main
----------------------------------------------


type TrainElem = ([LA.R 5], LA.R 5)
type Train = [TrainElem]


-- | The dataset consists of pairs (input, target output)
trainData :: Train
trainData =
--   [ ([zers, zers, zers, zers, zers], zers)
--   , ([zers, zers, zers, zers, ones], ones)
--   , ([zers, zers, zers, ones, zers], ones)
--   , ([zers, zers, ones, zers, zers], ones)
--   , ([zers, ones, zers, zers, ones], ones)
--   , ([ones, zers, zers, zers, zers], ones)
--   , ([ones, ones, ones, ones, ones], ones)
  [ ([], zers)
  , ([zers], zers)
  , ([ones], ones)
  , ([zers, zers], zers)
  , ([ones, zers], ones)
  , ([zers, ones], ones)
  , ([ones, ones], ones)
--   , ([zers, zers, zers], zers)
--   , ([ones, zers, zers], ones)
--   , ([zers, ones, zers], ones)
--   , ([ones, ones, zers], ones)
--   , ([zers, zers, ones], ones)
--   , ([ones, zers, ones], ones)
--   , ([zers, ones, ones], ones)
--   , ([ones, ones, ones], ones)
  ] where
    k = 5
    vals v = LA.vector . take k $ repeat v
    ones = vals 1
    zers = vals 0


main :: IO ()
main = do
  -- an actual network
  let list k = take k $ cycle [0, 1, 2]
      zero k = take k $ repeat 0
      ffn = FFN
        { _nWeights1 = LA.matrix $ zero (5*10)
        , _nBias1 = LA.vector $ zero 5
        , _nWeights2 = LA.matrix $ zero (5*5)
        , _nBias2 = LA.vector $ zero 5
        }
      rnn = RNN
        { _ffn = ffn
        , _h0 = LA.vector $ zero 5
        }
      -- input = [LA.vector . take 5 $ repeat 1]
      -- target = LA.vector [1, 2, 3, 4, 1]
  -- _rnn' <- gradDesc1 100 input target rnn
  rnn' <- gradDesc 1000 0.01 trainData rnn
  let ones = LA.vector . take 5 $ repeat 1
      zers = LA.vector . take 5 $ repeat 0
      resOnes = head $ runRNN (BP.auto rnn')
              (map BP.auto [ones, ones, ones, ones, ones])
      resZers = head $ runRNN (BP.auto rnn')
              (map BP.auto [zers, zers, zers, zers, zers])
      resMix1 = head $ runRNN (BP.auto rnn')
              (map BP.auto [ones, zers, zers, zers, zers, zers, zers, zers])
      resMix2 = head $ runRNN (BP.auto rnn')
              (map BP.auto [zers, zers, ones, zers, zers])
      resMix3 = head $ runRNN (BP.auto rnn')
              (map BP.auto [zers, zers, zers, zers, zers, zers, zers, ones])
  -- forM_ res $ \x -> print $ BP.evalBP0 x
  LBP.disp 5 $ BP.evalBP0 resOnes
  LBP.disp 5 $ BP.evalBP0 resZers
  LBP.disp 5 $ BP.evalBP0 resMix1
  LBP.disp 5 $ BP.evalBP0 resMix2
  LBP.disp 5 $ BP.evalBP0 resMix3
  return ()


----------------------------------------------
-- Rendering
----------------------------------------------


-- showVect :: (KnownNat n) => LA.R n -> String
-- showVect = undefined
