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

-------------------------
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE ScopedTypeVariables #-}
-------------------------


module Net.ArcGraph.Graph
  ( Graph(..)
  , Node(..)
  , Arc
  , nmap
  , amap
  , posTag

  , graphNodes
  , graphArcs
  , incoming
  , outgoing

  , New(..)
  , newMap

  , (:&) (..)
  , pattern (:&&)
  ) where


import           GHC.Generics (Generic)
import           GHC.TypeNats (KnownNat, natVal)

import           System.Random (randomRIO)

import           Control.DeepSeq (NFData)
import           Control.Lens (Lens)
import           Control.Monad (forM)

import           Data.Proxy (Proxy(..))
import           Data.Binary (Binary)
import qualified Data.Array as A
import qualified Data.Map.Strict as M
import qualified Data.Set as S
import qualified Data.Text as T
import qualified Data.Graph as G


import qualified Numeric.Backprop as BP
import           Numeric.Backprop (Backprop, Reifies, W, BVar, (^^.))
import qualified Numeric.LinearAlgebra.Static as LA
import           Numeric.LinearAlgebra.Static.Backprop (R, L)

import           Numeric.SGD.ParamSet (ParamSet)

import           Net.Basic
import qualified Net.FeedForward as FFN
import           Net.FeedForward (FFN(..))



----------------------------------------------
-- Graph
----------------------------------------------


-- | Local graph type
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


-- | Arc label mapping
amap :: (b -> c) -> Graph a b -> Graph a c
amap f g =
  g {arcLabelMap = fmap f (arcLabelMap g)}


-- | A graph arc (edge)
type Arc = (G.Vertex, G.Vertex)


-- | Structured node
data Node dim nlb = Node
  { nodeEmb :: R dim
    -- ^ Node embedding vector
  , nodeLab :: nlb
    -- ^ Node label (e.g., POS tag)
  , nodeLex :: T.Text
    -- ^ Lexical content (used for ,,unordered'' biaffinity)
  } deriving (Show, Binary, Generic)


-- | POS tag of the given vertex. 
posTag :: Graph (Node d a) b -> G.Vertex -> Maybe a
posTag g v = nodeLab <$> M.lookup v (nodeLabelMap g)


-- -- | Arc label of the given arc. 
-- arcLabel :: Graph (Node d a) b -> Arc -> Maybe a
-- arcLabel g (v, w) = M.lookup v (arcLabelMap g)


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
-- Type voodoo
----------------------------------------------


-- | Custom pair, with Backprop and ParamSet instances, and nice Backprop
-- pattern.
data a :& b = !a :& !b
  deriving (Show, Generic, Binary, NFData, Backprop, ParamSet)
infixr 2 :&

pattern (:&&) :: (Backprop a, Backprop b, Reifies z W)
              => BVar z a -> BVar z b -> BVar z (a :& b)
pattern x :&& y <- (\xy -> (xy ^^. t1, xy ^^. t2)->(x, y))
  where
    (:&&) = BP.isoVar2 (:&) (\case x :& y -> (x, y))
{-# COMPLETE (:&&) #-}


t1 :: Lens (a :& b) (a' :& b) a a'
t1 f (x :& y) = (:& y) <$> f x

t2 :: Lens (a :& b) (a :& b') b b'
t2 f (x :& y) = (x :&) <$> f y


----------------------------------------------
-- New
----------------------------------------------


class New a b p where
  new
    :: S.Set a
      -- ^ Set of node labels
    -> S.Set b
      -- ^ Set of arc labels
    -> IO p

instance New a b Double where
  new _ _ = randomRIO (-0.01, 0.01)

instance (KnownNat n) => New a b (R n) where
  new _ _ = LA.vector <$> randomList n
    where
      n = proxyVal (Proxy :: Proxy n)
      proxyVal = fromInteger . toInteger . natVal

instance (KnownNat n, KnownNat m) => New a b (L n m) where
  new _ _ = LA.matrix <$> randomList (n*m)
    where
      n = proxyVal (Proxy :: Proxy n)
      m = proxyVal (Proxy :: Proxy m)
      proxyVal = fromInteger . toInteger . natVal

instance (KnownNat i, KnownNat h, KnownNat o) => New a b (FFN i h o) where
  new xs ys = FFN
    <$> new xs ys
    <*> new xs ys
    <*> new xs ys
    <*> new xs ys

instance (New a b p1, New a b p2) => New a b (p1 :& p2) where
  new xs ys = do
    p1 <- new xs ys
    p2 <- new xs ys
    return (p1 :& p2)


-- | Create a new, random map.
newMap
  :: (Ord k, New a b v)
  => S.Set k
  -> S.Set a
  -> S.Set b
  -> IO (M.Map k v)
newMap keySet xs ys =
  fmap M.fromList .
    forM (S.toList keySet) $ \key -> do
      (key,) <$> new xs ys
