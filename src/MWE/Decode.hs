{-# LANGUAGE RecordWildCards #-}


-- | MWE decoding module.  The functionality implemented here corresponds to
-- decoding in the global encoding scheme described in the accompanying paper.


module MWE.Decode
  ( annotate
  ) where


import           Control.Monad (guard)

import qualified Data.Set as S
import qualified Data.Map.Strict as M
import qualified Data.Graph as G
import qualified Data.Foldable as Fold
import           Data.Semigroup (Max(..))

import qualified Format.Cupt as Cupt
import qualified Graph


-- | Annotate the sentence with the given MWE type, given the specified arc and
-- node labeling.
annotate
  :: Cupt.MweTyp
    -- ^ MWE type to annotate with
  -> Cupt.Sent
    -- ^ .cupt sentence to annotate
  -> Graph.Labeling Bool
    -- ^ Node/arc labeling
  -> Cupt.Sent
annotate mweTyp cupt Graph.Labeling{..} =

  map enrich cupt
  
  where

    -- Enrich the token with new MWE information
    enrich tok = Prelude.maybe tok id $ do
      Cupt.TokID i <- return (Cupt.tokID tok)
      mweId <- M.lookup i mweIdMap
      let newMwe = (mweId, mweTyp)
      return tok {Cupt.mwe = newMwe : Cupt.mwe tok}

    -- Determine the set of MWE nodes and arcs
    nodSet = trueKeys nodLab
    arcSet = trueKeys arcLab `S.union` nodeLoops

    -- The set of keys with `True` values
    trueKeys m = S.fromList $ do
      (x, val) <- M.toList m
      guard val
      return x

    -- The set of one node cycles
    nodeLoops = S.fromList $ do
      v <- S.toList nodSet
      return (v, v)

    -- Determine the mapping from nodes to new MWE id's
    ccs = findConnectedComponents arcSet
    mweIdMap = M.fromList . concat $ do
      (cc, mweId) <- zip ccs [maxMweID cupt + 1 ..]
      (v, w) <- S.toList cc
      return $ filter ((`S.member` nodSet) . fst)
        [(v, mweId), (w, mweId)]


-- | Given a set of graph arcs, determine all the connected arc subsets in the
-- corresponding graph.
findConnectedComponents
  :: S.Set Graph.Arc
  -> [S.Set Graph.Arc]
findConnectedComponents arcSet
  | S.null arcSet = []
  | otherwise
      -- Some components can be empty!  Perhaps because the graph is sparse?
      = filter (not . S.null)
      $ map (arcsInTree arcSet) (G.components graph)
  where
    vertices = S.toList (nodesIn arcSet)
    graph = G.buildG
      (minimum vertices, maximum vertices)
      (S.toList arcSet)


-- | Determine the set of arcs in the given connected graph component.
--
-- NOTE: This could be done much more efficiently!
--
arcsInTree
  :: S.Set Graph.Arc
    -- ^ The set of all arcs
  -> G.Tree G.Vertex
    -- ^ Connected component
  -> S.Set Graph.Arc
arcsInTree arcSet cc = S.fromList $ do
  (v, w) <- S.toList arcSet
  guard $ v `S.member` vset
  guard $ w `S.member` vset
  return (v, w)
  where
    vset = S.fromList (Fold.toList cc)


-- | The set of nodes in the given arc set.
nodesIn :: S.Set Graph.Arc -> S.Set G.Vertex
nodesIn arcSet =
  (S.fromList . concat)
    [[v, w] | (v, w) <- S.toList arcSet]


-- | Determine the maximum mwe ID present in the given sentence.
maxMweID :: Cupt.Sent -> Int
maxMweID =
  getMax . Fold.foldMap mweID
  where
    mweID tok = case Cupt.mwe tok of
      [] -> Max 0
      xs -> Max . maximum $ map fst xs
