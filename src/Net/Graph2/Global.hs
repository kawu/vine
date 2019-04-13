{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE Rank2Types #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveFunctor #-}


-- | Global labelling probability


module Net.Graph2.Global
  ( Labelling(..)
  , Version(..)
  , probLog
  ) where


import           GHC.Generics (Generic)

import           Control.Monad (guard)

import qualified Data.Map.Strict as M
import qualified Data.Graph as G
import qualified Data.List as List
import qualified Data.Maybe as Maybe

import           Dhall (Interpret)

import qualified Numeric.Backprop as BP
import           Numeric.LinearAlgebra.Static.Backprop
  (BVar, Reifies, W, dot)

import qualified Net.Graph2.BiComp as B
import           Net.Graph2.BiComp (Pot, Prob, Vec(..), Vec8, Out(..))
import           Graph (Graph, Arc, incoming, outgoing)
import qualified Graph
import qualified Net.Graph2.Marginals as Margs


----------------------------------------------
-- Probability
----------------------------------------------


-- | If you pick `Contrained`, only the labellings satisfying certain
-- properties will be considered in the calculation of the partition factor.
-- Otherwise, the entire set of the internally consistent labellings will be
-- considered.
data Version
  = Free
  | Constrained
  deriving (Generic)

instance Interpret Version


-- | Probability of graph compound labelling
probLog
  :: (Reifies s W)
  => Version
    -- ^ Constrained version or not?
  -> Graph a b
    -- ^ The underlying graph
  -> Labelling Bool
    -- ^ Target labelling of which we want to determine the probability
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ The corresponding arc potential map (normal domain!)
  -> M.Map G.Vertex (BVar s Double)
    -- ^ The corresponding node potential map (normal domain!)
  -> BVar s Double
    -- ^ The resulting (sum of) potential(s) (log domain!)
probLog version graph ell potMap nodMap =
  score graph ell potMap nodMap - partition
  where
    root = treeRoot graph
    partition =
      case version of
        Constrained -> (sumLog . Maybe.mapMaybe Margs.unMVar)
          [inside root False False, inside root True False]
          where
            -- Funny thing is, the inside calculation blows up without
            -- memoization!
            inside = Margs.insideLogMemoC graph potMap nodMap
        Free ->
          inside root False `addLog` inside root True
          where
            inside = Margs.insideLogMemo graph potMap nodMap


-- | Global score of the given labelling
score
  :: (Reifies s W)
  => Graph a b
    -- ^ The underlying graph
  -> Labelling Bool
    -- ^ Target labelling of which we want to determine the probability
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ The corresponding arc potential map (normal domain!)
  -> M.Map G.Vertex (BVar s Double)
    -- ^ The corresponding node potential map (normal domain!)
  -> BVar s Double
score graph Labelling{..} potMap nodMap =
  arcScore + nodeScore
  where
    arcScore = sum $ do
      (v, w) <- Graph.graphArcs graph
      let out = B.Out
            { arcVal = arcLab M.! (v, w)
            , depVal = nodLab M.! v
            , hedVal = nodLab M.! w
        }
      return $ arcPot potMap (v, w) out
    nodeScore = sum $ do
      v <- Graph.graphNodes graph
      -- v's score is 0 if v is labelled with 0
      guard $ nodLab M.! v
      return $ nodMap M.! v


----------------------------------------------
-- Labelling
--
-- (copy from Net.Graph2)
----------------------------------------------


-- | Graph node/arc labeling
data Labelling a = Labelling
  { nodLab :: M.Map G.Vertex a
  , arcLab :: M.Map Arc a
  } deriving (Functor)

instance Semigroup a => Semigroup (Labelling a) where
  Labelling n1 a1 <> Labelling n2 a2 =
    Labelling (n1 <> n2) (a1 <> a2)

instance Monoid a => Monoid (Labelling a) where
  mempty = Labelling M.empty M.empty


----------------------------------------------
-- Utils
--
-- (copy from marginals and net/graph2)
----------------------------------------------


-- | Potential of the given arc
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
thereLog
  :: (Reifies s W) 
  => M.Map G.Vertex (BVar s Double)
  -> G.Vertex
  -> Bool
  -> BVar s Double
thereLog nodMap u z
  | z = nodMap M.! u
  | otherwise = 0.0


-- | The list of possible values of an arc
arcValues :: [Out Bool]
arcValues = B.enumerate


-- | Retrieve the root of the given directed graph/tree.  It is assumed that
-- each arc points in the direction of the parent in the tree.  The function
-- raises an error if the input graph is not a well-formed tree.
treeRoot :: Graph a b -> G.Vertex
treeRoot g =
  case roots of
    [v] -> v
    [] -> error "Graph2.treeRoot: no root found!"
    _ -> error "Graph2.treeRoot: several roots found!"
  where
    roots = do
      v <- Graph.graphNodes g
      guard . null $ Graph.outgoing v g
      return v

----------------------------------------------
-- Basic operations in log domain
--
-- (copy from marginals)
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
