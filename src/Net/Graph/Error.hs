{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE NoMonomorphismRestriction #-}


-- | Generic error calculation functions


module Net.Graph.Error
  ( softMaxCrossEntropy
  , softMaxCrossEntropy'
  , errorOne
  , errorMany
  ) where


import           Prelude hiding (or)

import           GHC.TypeNats (KnownNat)

import qualified Data.List as List
import qualified Data.Map.Strict as M

import qualified Numeric.LinearAlgebra as LD
import qualified Numeric.LinearAlgebra.Static as LA

import qualified Numeric.Backprop as BP
import           Numeric.LinearAlgebra.Static.Backprop
  (R, BVar, Reifies, W)

import qualified Net.Graph.Arc as Arc
import           Net.Graph.Arc (Pot, Prob, Vec(..), Vec8)


-- | Softmax + cross-entropy with manual gradient calculation.  Type-safe,
-- higher-level variant of `softMaxCrossEntropy'`.
softMaxCrossEntropy
  :: forall s. (Reifies s W)
  => Vec8 Prob
    -- ^ Target ,,true'' distribution
  -> BVar s (Vec8 Pot)
    -- ^ Output ,,artificial'' distribution (represted by potentials)
  -> BVar s Double
softMaxCrossEntropy p0 q0 =
  softMaxCrossEntropy' (Arc.unVec p0) (BP.coerceVar q0)
--   checkNaNBP "softMaxCrossEntropy" $ negate (p `dot` LBP.vmap' log' q)
--   where
--     -- p = BP.coerceVar p0 :: BVar s (R 8)
--     p = BP.coerceVar (BP.auto p0) :: BVar s (R 8)
--     q = BP.coerceVar q0
--     -- avoid NaN when p = 0 and q = 0
--     log' x
--       | x > 0 = log x
--       | otherwise = -1.0e8


-- | Softmax + cross-entropy with manual gradient calculation.  Code adapted
-- from the backprop tutorial to use the exp-normalize trick (see
-- https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick).
softMaxCrossEntropy'
  :: (KnownNat n, Reifies s W)
  => R n
    -- ^ Target *probabilities*
  -> BVar s (R n)
    -- ^ Output *potentials*
  -> BVar s Double
softMaxCrossEntropy' x = BP.liftOp1 . BP.op1 $ \y ->
  let ymax     = LD.maxElement (LA.extract y)
      expy     = LA.dvmap (exp . (\o -> o - ymax)) y
      toty     = LD.sumElements (LA.extract expy)
      softMaxY = LA.konst (1 / toty) * expy
      smce     = negate (LA.dvmap log' softMaxY `LA.dot` x)
--    in trace ("y: " ++ show (Util.toList y)) $
--       trace ("ymax: " ++ show ymax) $
--       trace ("expy: " ++ show (Util.toList expy)) $
--       trace ("toty: " ++ show toty) $
--       trace ("softMaxY: " ++ show (Util.toList softMaxY)) $
--       trace ("smce: " ++ show smce) $
   in ( smce
      , \d -> LA.konst d * (softMaxY - x)
      )
  where
    -- to avoid NaN when p_i = 0 and q_i = 0
    log' v
      | v > 0 = log v
      -- TODO: is it a proper choice of epsilon?
      | otherwise = -1.0e10


-- | Calculate the error between the target probabilities and the output scores
-- in the given pair of maps, where the values corresponding to each other are
-- assigned to the same key.
--
-- NOTE: The two maps must have the same set of keys, which is currently not
-- checked.
--
-- TODO: Make sure that the two maps have the same set of keys?
--
errorOne
  :: (Ord a, Reifies s W)
  => M.Map a (Vec8 Prob)
    -- ^ Target probabilities
  -> M.Map a (BVar s (Vec8 Pot))
    -- ^ Output scores
  -> BVar s Double
errorOne target output = sum $ do
  (key, tval) <- M.toList target
  let oval = output M.! key
  return $ softMaxCrossEntropy tval oval


-- -- | A `BVar` lifted sum of elements.
-- liftedSum :: (Reifies s W) => [BVar s Double] -> BVar s Double
-- liftedSum = List.foldl' (+) 0


-- | Extension of `errorOne` to a pair of lists
errorMany
  :: (Ord a, Reifies s W)
  => [M.Map a (Vec8 Prob)] 
    -- ^ Targets
  -> [M.Map a (BVar s (Vec8 Pot))] 
    -- ^ Outputs
  -> BVar s Double
errorMany targets outputs =
  go targets outputs
  where
    go ts os =
      case (ts, os) of
        (t:tr, o:or) -> errorOne t o + go tr or
        ([], []) -> 0
        _ -> error "errorMany: lists of different size"
