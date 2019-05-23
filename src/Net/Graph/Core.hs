{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE Rank2Types #-}
{-# LANGUAGE FlexibleContexts #-}


-- | The module provides core graph network functionality.


module Net.Graph.Core
  (
  -- * Types
    Version (..)

  -- * Local scores
  , arcValues
  , arcPot
  , thereLog

  -- * Ops in log domain
  , mulLog
  , addLog
  , sumLog
  , productLog
  ) where


import           GHC.Generics (Generic)

import qualified Data.List as List
import qualified Data.Graph as G
import qualified Data.Map.Strict as M

import           Dhall (Interpret)

import qualified Numeric.Backprop as BP
import           Numeric.LinearAlgebra.Static.Backprop (BVar, Reifies, W, dot)

import           Net.Graph.Arc (Out, Vec8, Pot)
import qualified Net.Graph.Arc as B
import           Graph (Arc)


----------------------------------------------
-- Types
----------------------------------------------


-- | If you pick `Contrained`, only the labellings satisfying certain
-- properties will be considered in the calculation of the partition factor.
-- Otherwise, the entire set of the internally consistent labellings will be
-- considered.
data Version
  = Free
  | Constrained
  | Local
    -- ^ The `Local` version is somewhat special. It only makes sense within
    -- the context of marginal probabilities, where it means calculating them
    -- in complete separation from other arcs and nodes.  Experimental feature!
  deriving (Generic)

instance Interpret Version


----------------------------------------------
-- Local scoring
----------------------------------------------


-- | The list of possible values of an arc
arcValues :: [Out Bool]
arcValues = B.enumerate


-- | Score of the given complex arc label.
--
-- TODO: Rename as `arcScore`?
--
arcPot
  :: forall s. (Reifies s W)
  => M.Map Arc (BVar s (Vec8 Pot))
    -- ^ Potential map
  -> Arc
    -- ^ The arc in question
  -> Out Bool
    -- ^ The value
  -> BVar s Double
arcPot potMap arc out =
  potVec `dot` BP.constVar (B.mask out)
  where
    potVec = BP.coerceVar (potMap M.! arc)
{-# INLINE arcPot #-}


-- | Node potential at a given node in log domain
--
-- TODO: rename as `nodeScore`?  No need to specify that this is in log domain,
-- the score is in log domain by default, right?
--
thereLog
  :: (Reifies s W) 
  => M.Map G.Vertex (BVar s Double)
  -> G.Vertex
  -> Bool
  -> BVar s Double
thereLog nodMap u z
  | z = nodMap M.! u
  | otherwise = 0.0


-- -- | (Exp-)potential of the given arc  (exp . arcPot)
-- arcPotExp
--   :: forall s. (Reifies s W)
--   => M.Map Arc (BVar s (Vec8 Pot))
--     -- ^ Potential map
--   -> Arc
--     -- ^ The arc in question
--   -> Out Bool
--     -- ^ The value
--   -> BVar s Double
-- arcPotExp potMap arc =
--   exp . arcPot potMap arc


-- -- | Node potential at a given node (TODO: make analogy to `arcPotExp`)
-- there
--   :: (Reifies s W) 
--   => M.Map G.Vertex (BVar s Double)
--   -> G.Vertex
--   -> Bool
--   -> BVar s Double
-- there nodMap u z
--   | z = exp (nodMap M.! u)
--   | otherwise = 1.0


----------------------------------------------
-- Basic operations in log domain
----------------------------------------------


-- | Multiplication (log domain)
mulLog :: Floating a => a -> a -> a
mulLog = (+)
{-# INLINE mulLog #-}


-- | Addition (log domain)
-- TODO: provide something more efficient?
addLog :: (Floating a, Ord a) => a -> a -> a
addLog x y = sumLog [x, y]
{-# INLINE addLog #-}


-- -- | Sum (log domain)
-- sumLog :: Floating a => [a] -> a
-- sumLog = log . sum . map exp
-- {-# INLINE sumLog #-}


-- | Sum (log domain)
sumLog :: (Floating a, Ord a) => [a] -> a
sumLog xs =
  -- Code copied from the log-float library
  theMax + log theSum
  where
    theMax = maximum xs
    theSum = List.foldl' (\acc x -> acc + exp (x - theMax)) 0 xs
{-# INLINE sumLog #-}


-- | Product (log domain)
productLog :: Floating a => [a] -> a
productLog = sum
{-# INLINE productLog #-}
