{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DeriveGeneric #-}


module Test
  ( -- trainTest
  ) where



import           GHC.Generics (Generic)
import           Lens.Micro.TH (makeLenses)
import qualified Numeric.Backprop as BP
import           Numeric.Backprop ((^^.), BVar, Reifies, W)

import qualified Numeric.SGD as SGD
import qualified Numeric.SGD.Momentum as Mom


-----------------------------------------------------
-- Polynomial
-----------------------------------------------------


-- | Polynomial: a0 + a1*x + a2*x^2
data Poly = Poly
  { _a0 :: Double
  , _a1 :: Double
  , _a2 :: Double
  }
  deriving (Show, Generic)

instance BP.Backprop Poly
instance SGD.ParamSet Poly
makeLenses ''Poly


-- | Evaluate polynomial `p` on argument `x`
run :: Reifies s W => BVar s Poly -> BVar s Double -> BVar s Double
run p x =
  (p ^^. a0) +
  (p ^^. a1) * x +
  (p ^^. a2) * x*x


-- -----------------------------------------------------
-- -- Error calculation
-- -----------------------------------------------------
-- 
-- 
-- -- | Squared error between `xs` and `ys`
-- squaredError :: Floating a => [a] -> [a] -> a
-- squaredError xs ys = sum $ do
--   (x, y) <- zip xs ys
--   return $ (x-y)**2
-- 
-- 
-- -- | Error the polynomial yields on a given dataset
-- polyError
--   :: Reifies s W
--   => [(Double, Double)]
--   -> BVar s Poly
--   -> BVar s Double
-- polyError dataSet poly =
--   squaredError
--     (map (run poly) input)
--     target
--   where
--     input  = map (BP.constVar . fst) dataSet
--     target = map (BP.constVar . snd) dataSet
-- 
-- 
-- -----------------------------------------------------
-- -- Training
-- -----------------------------------------------------
-- 
-- 
-- -- | Small training dataset: pairs of input and target values
-- trainData :: [(Double, Double)]
-- trainData =
--   [ (-2,  3)
--   , ( 0,  2)
--   , ( 1, -1)
--   -- , ( 2, -2)
--   ]
-- 
-- 
-- -- | Objective function
-- trainTest :: IO Poly
-- trainTest = do
--   SGD.withVect trainData $ \dataSet ->
--     SGD.sgd sgdCfg dataSet gradient quality
--       (Poly 0 0 0)
--   where
--     gradient xs = BP.gradBP (polyError xs)
--     quality x = BP.evalBP (polyError [x])
--     sgdCfg = SGD.Config
--       { SGD.iterNum = 1000
--       , SGD.batchSize = 1
--       , SGD.batchRandom = False
-- --       , SGD.method = SGD.AdaDelta
-- --           { SGD.decay = 0.9
-- --           , SGD.eps = 1.0e-6
-- --           }
--       , SGD.method = SGD.Momentum $ Mom.Config
--           { Mom.gamma = 0.9
--           , Mom.gain0 = 0.01
--           , Mom.tau = 500
--           }
--       , SGD.reportEvery = 100
--       }
