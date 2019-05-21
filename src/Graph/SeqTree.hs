
-- | The module provides functions which rely on some additional assumptions as
-- to the structore of the (dependency) graph we work with.  Namely:
--
-- * Each arc points in the direction of its parent in the tree
-- * The sequential order of the nodes is consistent with their
--   node identifiers (see `onLeft` and `onRight`)


module Graph.SeqTree
  ( hasCycles
  -- , isForest
  , roots
  , treeRoot
  , onLeft
  , onRight
  ) where


import           Control.Monad (guard)

import qualified Data.Array as A
import qualified Data.Graph as G
import qualified Data.Map.Strict as M

import           Graph


-- | Are there any cycles in the given graph?
hasCycles :: Graph a b -> Bool
hasCycles g =
  any isCyclic sccs
  where
    sccs = G.stronglyConnComp
      [(x, x, ys) | (x, ys) <- A.assocs (graphStr g)]
    isCyclic (G.CyclicSCC _) = True
    isCyclic _ = False


-- -- | A graph is a forest if it has no cycles.
-- isForest :: Graph a b -> Bool
-- isForest = not . hasCycles


-- | Retrieve the list of roots of the given directed graph/tree.  It is
-- assumed that each arc points in the direction of the parent in the tree.
-- The function raises an error if the input graph has cycles.
roots :: Graph a b -> [G.Vertex]
roots g
  | hasCycles g = error "Graph.roots: graph has cycles"
  | otherwise = do
      v <- Graph.graphNodes g
      guard . null $ Graph.outgoing v g
      return v


-- | Retrieve the root of the given directed graph/tree.  It is assumed that
-- each arc points in the direction of the parent in the tree.  The function
-- raises an error if the input graph is not a well-formed tree.
treeRoot :: Graph a b -> G.Vertex
treeRoot g =
  case roots g of
    [v] -> v
    [] -> error "Graph.treeRoot: no root found!"
    _ -> error "Graph.treeRoot: several roots found!"


-- | Extract the sister node directly on the left (if any).
--
-- TODO: assumes a particular tree structure.
--
onLeft :: G.Vertex -> Graph a b -> Maybe G.Vertex
onLeft v g = fst <$> M.lookupLT v (nodeLabelMap g)


-- | Extract the sister node directly on the right (if any).
--
-- TODO: assumes a particular tree structure.
--
onRight :: G.Vertex -> Graph a b -> Maybe G.Vertex
onRight v g = fst <$> M.lookupGT v (nodeLabelMap g)
