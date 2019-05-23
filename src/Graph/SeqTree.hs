-- | The module provides functions which rely on some additional assumptions as
-- to the structore of the (dependency) graph we work with.  Namely:
--
-- * Each arc points in the direction of its parent in the tree
-- * The sequential order of the nodes is consistent with their
--   node identifiers (see `onLeft` and `onRight`)


module Graph.SeqTree
  ( hasCycles
  , roots
  , treeRoot
  , onLeft
  , onRight
  , treeConnectAll
  ) where


import           Control.Monad (guard)

import qualified Data.List as List
import qualified Data.Array as A
import qualified Data.Graph as G
import qualified Data.Map.Strict as M
import qualified Data.Set as S

import           Graph


----------------------------------------------
-- Various
----------------------------------------------


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
onLeft :: G.Vertex -> Graph a b -> Maybe G.Vertex
onLeft v g = fst <$> M.lookupLT v (nodeLabelMap g)


-- | Extract the sister node directly on the right (if any).
onRight :: G.Vertex -> Graph a b -> Maybe G.Vertex
onRight v g = fst <$> M.lookupGT v (nodeLabelMap g)


----------------------------------------------
-- Connecting vertices
----------------------------------------------


-- | Determine the set of arcs that allows to connect the given set of
-- vertices.
treeConnectAll
  :: Graph.Graph a b
  -> S.Set G.Vertex
  -> S.Set Graph.Arc
treeConnectAll graph =
  go . S.toList
  where
    go (v:w:us) = treeConnect graph v w `S.union` go (w:us)
    go _ = S.empty


-- | Determine the set of arcs that allows to connect the two vertices.
treeConnect
  :: Graph.Graph a b
  -> G.Vertex
  -> G.Vertex
  -> S.Set Graph.Arc
treeConnect graph v w =
  arcSet v u `S.union` arcSet w u
  where
    arcSet x y = (S.fromList . pathAsArcs) (treePath graph x y)
    u = commonAncestor graph v w


-- | Find tha path from the first to the second vertex.
treePath :: Graph.Graph a b -> G.Vertex -> G.Vertex -> [G.Vertex]
treePath graph v w =
  List.takeWhile (/=w) (pathToRoot graph v) ++ [w]


-- | Convert the list of vertices to a list of arcs on the path.
pathAsArcs :: [G.Vertex] -> [Graph.Arc]
pathAsArcs (x:y:xs) = (x, y) : pathAsArcs (y:xs)
pathAsArcs _ = []


-- | Commmon ancestor of the two given nodes (in a forest)
commonAncestor
  :: Graph.Graph a b
  -> G.Vertex
  -> G.Vertex
  -> G.Vertex
commonAncestor graph v w =
  firstCommonElem
    (pathToRoot graph v)
    (pathToRoot graph w)


-- | Find the first common element of the given lists.
firstCommonElem :: Eq a => [a] -> [a] -> a
firstCommonElem xs ys
  = fst . safeHead . reverse
  . List.takeWhile (uncurry (==))
  $ zip (reverse xs) (reverse ys)
  where
    safeHead (e:_) = e
    safeHead [] =
      error "Graph.firstCommonElem: no common element found"


-- | Find the path from the given node to a tree root.  The given graph must be
-- a forest.
pathToRoot :: Graph.Graph a b -> G.Vertex -> [G.Vertex]
pathToRoot graph =
  go
  where
    go v =
      case Graph.outgoing v graph of
        [] -> [v]
        [w] -> v : go w
        _ -> error "Graph.pathToRoot: the given graph is not a tree/forest"
