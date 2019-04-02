{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE PolyKinds #-}


-- | The module provides the representation of a graph as it is used for VMWE
-- identification.


module Graph
  ( Graph(..)
  , Arc
  , nmap
  , nmap'
  , amap
  , nodeLabAt
  , arcLabAt

  , isAsc
  , mkAsc

  , graphNodes
  , graphArcs
  , incoming
  , outgoing
  ) where


import           GHC.Generics (Generic)

import           System.Random (randomRIO)

import           Control.Monad (forM)

import           Data.Proxy (Proxy(..))
import           Data.Binary (Binary)
import qualified Data.List as L
import qualified Data.Array as A
import qualified Data.Map.Strict as M
import qualified Data.Set as S
import qualified Data.Text as T
import qualified Data.Graph as G


import qualified Numeric.LinearAlgebra.Static as LA
import           Numeric.LinearAlgebra.Static.Backprop (R, L)

-- import           Net.Util
-- import qualified Net.FeedForward as FFN
-- import           Net.FeedForward (FFN(..))



----------------------------------------------
-- Graph
----------------------------------------------


-- | Our custom graph type
data Graph a b = Graph
  { graphStr :: G.Graph
    -- ^ The underlying directed graph
  , graphInv :: G.Graph
    -- ^ Inversed (transposed) `graphStr`
  , nodeLabelMap :: M.Map G.Vertex a
    -- ^ Label assigned to a given vertex
  , arcLabelMap :: M.Map Arc b
    -- ^ Label assigned to a given arc
  } deriving (Show, Eq, Ord, Generic, Binary)


-- | Node label mapping
nmap :: (a -> c) -> Graph a b -> Graph c b
nmap f g =
  g {nodeLabelMap = fmap f (nodeLabelMap g)}


-- | Node label mapping with node IDs
nmap' :: (G.Vertex -> a -> c) -> Graph a b -> Graph c b
nmap' f g =
  g {nodeLabelMap = M.mapWithKey f (nodeLabelMap g)}


-- | Arc label mapping
amap :: (b -> c) -> Graph a b -> Graph a c
amap f g =
  g {arcLabelMap = fmap f (arcLabelMap g)}


-- | A graph arc (edge)
type Arc = (G.Vertex, G.Vertex)


-- | Node label of the given vertex.
nodeLabAt :: Graph a b -> G.Vertex -> Maybe a
nodeLabAt g v = M.lookup v (nodeLabelMap g)


-- | Arc label of the given arc. 
arcLabAt :: Graph a b -> Arc -> Maybe b
arcLabAt g e = M.lookup e (arcLabelMap g)


----------------------------------------------
-- Graph Utils
----------------------------------------------


-- | Return the list of vertives in the graph.
graphNodes :: Graph a b -> [G.Vertex]
graphNodes = G.vertices . graphStr


-- | Return the list of vertives in the graph.
graphArcs :: Graph a b -> [Arc]
graphArcs = G.edges . graphStr


-- | Return the list of outgoing vertices.
outgoing :: G.Vertex -> Graph a b -> [G.Vertex]
outgoing v Graph{..} =
  graphStr A.! v


-- | Return the list of incoming vertices.
incoming :: G.Vertex -> Graph a b -> [G.Vertex]
incoming v Graph{..} =
  graphInv A.! v


----------------------------------------------
-- Graph monotonicity
----------------------------------------------


-- | Check if the incoming/outgoing lists are ascending.
mkAsc :: Graph a b -> Graph a b
mkAsc g = g
  { graphStr = _mkAsc $ graphStr g
  , graphInv = _mkAsc $ graphInv g
  }


-- | Check if the incoming/outgoing lists are ascending.
_mkAsc :: G.Graph -> G.Graph
_mkAsc = fmap L.sort


-- | Check if the incoming/outgoing lists are ascending.
isAsc :: Graph a b -> Bool
isAsc g = _isAsc (graphStr g) && _isAsc (graphInv g)


-- | Check if the incoming/outgoing lists are ascending.
_isAsc :: G.Graph -> Bool
_isAsc g = and $ do
  v <- G.vertices g
  return $ isAscList (g A.! v)


-- | Is the given list ascending?
isAscList :: (Ord a) => [a] -> Bool
isAscList (x:y:zs) = x < y && isAscList (y:zs)
isAscList _ = True
