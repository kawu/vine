{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}


-- | Global scoring


module Net.Graph.Global
  ( probLog
  ) where


import           Control.Monad (guard)

import qualified Data.Map.Strict as M
import qualified Data.Graph as G
import qualified Data.Maybe as Maybe

import           Numeric.LinearAlgebra.Static.Backprop (BVar, Reifies, W)

import           Net.Graph.Core
import qualified Net.Graph.Arc as B
import           Net.Graph.Arc (Pot, Vec8, Out(..))
import           Graph (Graph, Arc, Labeling(..))
import qualified Graph
import           Graph.SeqTree (treeRoot)
import qualified Net.Graph.Marginals as Margs


----------------------------------------------
-- Probability
----------------------------------------------


-- | Probability of graph compound labelling (log domain)
probLog
  :: (Reifies s W)
  => Version
    -- ^ Constrained version or not?
  -> Graph a b
    -- ^ The underlying graph
  -> Labeling Bool
    -- ^ Target labelling of which we want to determine the probability
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ The corresponding arc potential map (normal domain!)
  -> M.Map G.Vertex (BVar s Double)
    -- ^ The corresponding node potential map (normal domain!)
  -> BVar s Double
probLog version graph ell potMap nodMap =
  score graph ell potMap nodMap - partition
  where
    root = treeRoot graph
    partition =
      case version of
        Constrained -> (sumLog . Maybe.mapMaybe Margs.unMVar)
          [inside root False False, inside root True False]
          where
            -- Interesting thing is, the inside calculation blows up without
            -- memoization.  This is because it is used in a backpropagable
            -- manner.
            inside = Margs.insideLogMemoC graph potMap nodMap
        Free ->
          inside root False `addLog` inside root True
          where
            inside = Margs.insideLogMemo graph potMap nodMap
        Local ->
          -- Local doesn't make sense here, it can be only used within the
          -- context of marginal probabilities
          error "Global.probLog: cannot handle `Local`"


-- | Global score of the given labelling
score
  :: (Reifies s W)
  => Graph a b
    -- ^ The underlying graph
  -> Labeling Bool
    -- ^ Target labelling of which we want to determine the probability
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ The corresponding arc potential map (normal domain!)
  -> M.Map G.Vertex (BVar s Double)
    -- ^ The corresponding node potential map (normal domain!)
  -> BVar s Double
score graph Labeling{..} potMap nodMap =
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
