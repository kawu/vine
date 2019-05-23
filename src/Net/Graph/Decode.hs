{-# LANGUAGE RecordWildCards #-}


module Net.Graph.Decode
  ( treeTagGlobal
  , treeTagConstrained
  ) where


import qualified Data.Graph as G
import qualified Data.Map.Strict as M
import qualified Data.List as List
import           Data.Monoid (Any(..))

import           Graph
import           Graph.SeqTree

import           Net.Graph.Core (Labelling(..))
import           Net.Graph.Arc (Pot, Prob, Vec(..), Vec8, Out(..))
import qualified Net.Graph.Arc as Arc


----------------------------------------------
-- Global decoding
----------------------------------------------


-- | Determine the node/arc labeling which maximizes the global potential over
-- the given tree and return the resulting arc labeling.
--
-- WARNING: This function is only guaranteed to work correctly if the argument
-- graph is a tree!
--
treeTagGlobal
  :: Graph a b
  -> M.Map Arc (Vec8 Pot)
  -> M.Map G.Vertex Double
  -> Labelling Bool
treeTagGlobal graph labMap nodMap =
  let (trueBest, falseBest) =
        tagSubTree
          (treeRoot graph)
          graph
          (fmap Arc.explicate labMap)
          nodMap
      best = better trueBest falseBest
   in fmap getAny (bestLab best)


-- | The function returns two `Best`s:
--
--   * The best labeling if the label of the root is `True`
--   * The best labeling if the label of the root is `False`
--
tagSubTree
  :: G.Vertex
    -- ^ Root of the subtree
  -> Graph a b
    -- ^ The entire graph
  -> M.Map Arc (M.Map (Out Bool) Double)
    -- ^ Explicated labeling potentials
  -> M.Map G.Vertex Double
    -- ^ Node labeling potentials
  -> (Best, Best)
tagSubTree root graph lmap nmap =
  (bestWith True, bestWith False)
  where
    nodePot rootVal
      | rootVal = nmap M.! root
      | otherwise = 0.0
    bestWith rootVal = addNode root rootVal (nodePot rootVal) . mconcat $ do
      child <- Graph.incoming root graph
      let arc = (child, root)
          pot arcv depv = (lmap M.! arc) M.!
            Out {arcVal=arcv, hedVal=rootVal, depVal=depv}
          (true, false) = tagSubTree child graph lmap nmap
      return $ List.foldl1' better
        [ addArc arc True  (pot True  True)  true
        , addArc arc False (pot False True)  true
        , addArc arc True  (pot True  False) false
        , addArc arc False (pot False False) false ]


----------------------------------------------
-- Best (global decoding)
----------------------------------------------


-- | The best arc labeling for a given subtree.
data Best = Best
  { bestLab :: Labelling Any
    -- ^ Labelling (using `Any` guarantees that disjunction is used in case some
    -- label is accidentally assigned to a given object twice)
  , bestPot :: Double
    -- ^ Total potential
  }

instance Semigroup Best where
  Best l1 p1 <> Best l2 p2 =
    Best (l1 <> l2) (p1 + p2)

instance Monoid Best where
  mempty = Best mempty 0


-- | Impossible labeling (with infinitely poor potential)
impossible :: Best
impossible = Best mempty (read "-Infinity")


-- | Choose the better `Best`.
better :: Best -> Best -> Best
better b1 b2
  | bestPot b1 >= bestPot b2 = b1
  | otherwise = b2


-- | Add the given arc, its labeling, and the resulting potential to the given
-- `Best` structure.
addArc :: Arc -> Bool -> Double -> Best -> Best
addArc arc lab pot Best{..} = Best
  { bestLab = bestLab
      {arcLab = M.insert arc (Any lab) (arcLab bestLab)}
  , bestPot = bestPot + pot 
  }


-- | Set label of the given node in the given `Best` structure.  Similar to
-- `addArc`, but used when the potential of the node has been already accounted
-- for.
setNode :: G.Vertex -> Bool -> Best -> Best
setNode node lab best@Best{..} = best
  { bestLab = bestLab
      {nodLab = M.insert node (Any lab) (nodLab bestLab)}
  }


-- | Add the given node, its labeling, and the resulting potential to the given
-- `Best` structure.
addNode :: G.Vertex -> Bool -> Double -> Best -> Best
addNode node lab pot Best{..} = Best
  { bestLab = bestLab
      {nodLab = M.insert node (Any lab) (nodLab bestLab)}
  , bestPot = bestPot + pot
  }


----------------------------------------------
-- Constrained decoding'
----------------------------------------------


-- | Constrained version of `treeTagGlobal`
treeTagConstrained
  :: Graph a b
  -> M.Map Arc (Vec8 Pot)
  -> M.Map G.Vertex Double
  -> Labelling Bool
treeTagConstrained graph labMap nodMap =
  let Best4{..} =
        tagSubTreeC
          (treeRoot graph)
          graph
          (fmap Arc.explicate labMap)
          nodMap
      best = List.foldl1' better
        -- NOTE: `falseZeroOne` can be excluded in constrained decoding
        [true, falseZeroTrue, falseMoreTrue]
   in getAny <$> bestLab best


-- | Calculate `Best3` of the subtree.
tagSubTreeC
  :: G.Vertex
    -- ^ Root of the subtree
  -> Graph a b
    -- ^ The entire graph (tree)
  -> M.Map Arc (M.Map (Out Bool) Double)
    -- ^ Explicated labeling potentials
  -> M.Map G.Vertex Double
    -- ^ Node labeling potentials
  -> Best4
tagSubTreeC root graph lmap nmap =
  List.foldl' (<>)
    (emptyBest4 root $ nmap M.! root)
    (map bestFor children)
  where
    children = Graph.incoming root graph
    bestFor child =
      let arc = (child, root)
          pot arcv hedv depv = (lmap M.! arc) M.!
            Out {arcVal=arcv, hedVal=hedv, depVal=depv}
          Best4{..} = tagSubTreeC child graph lmap nmap
          -- NOTE: some of the configurations below are not allowed in
          -- constrained decoding and hence are commented out.
          true' = List.foldl1' better
            [ addArc arc True  (pot True  True True)  true
            , addArc arc False (pot False True True)  true
            -- , addArc arc True  (pot True  True False) falseZeroTrue
            , addArc arc False (pot False True False) falseZeroTrue
            , addArc arc True  (pot True  True False) falseOneTrue
            -- , addArc arc False (pot False True False) falseOneTrue
            , addArc arc True  (pot True  True False) falseMoreTrue
            , addArc arc False (pot False True False) falseMoreTrue
            ]
          falseZeroTrue' = List.foldl1' better
            [ addArc arc False (pot False False True)  true
            , addArc arc False (pot False False False) falseZeroTrue
            -- , addArc arc False (pot False False False) falseOneTrue
            , addArc arc False (pot False False False) falseMoreTrue
            ]
          falseOneTrue' = List.foldl1' better
            [ addArc arc True (pot True False True)  true
            -- , addArc arc True (pot True False False) falseZeroTrue
            , addArc arc True (pot True False False) falseOneTrue
            , addArc arc True (pot True False False) falseMoreTrue
            ]
       in Best4
            { true = true'
            , falseZeroTrue = falseZeroTrue'
            , falseOneTrue  = falseOneTrue'
            , falseMoreTrue = impossible
            }


----------------------------------------------
-- Best4 (constrained decoding)
----------------------------------------------


-- | The best labeling
data Best4 = Best4
  { true          :: Best
    -- ^ The label of the root is `True`.  The root's outgoing arc can be
    -- `True` or `False.
  , falseZeroTrue :: Best
    -- ^ The label of the root is `False` and all its incoming arcs are `False`
    -- too.  The outgoing arc must be `False`.
  , falseOneTrue  :: Best
    -- ^ The label of the root is `False` and exactly one of its incoming arcs
    -- is `True`.  The outgoing arc must be `True`.
  , falseMoreTrue :: Best
    -- ^ The label of the root is `False` and more than one of its incoming
    -- arcs is `True`.  The outgoing arc can be `True` or `False.
  }

instance Semigroup Best4 where
  b1 <> b2 = Best4
    { true =
        true b1 <> true b2
    , falseZeroTrue =
        falseZeroTrue b1 <> falseZeroTrue b2
    , falseOneTrue = List.foldl1' better
        [ falseZeroTrue b1 <> falseOneTrue  b2
        , falseOneTrue  b1 <> falseZeroTrue b2
        ]
    , falseMoreTrue = List.foldl1' better
        [ falseZeroTrue b1 <> falseMoreTrue b2
        , falseMoreTrue b1 <> falseZeroTrue b2
        , falseOneTrue  b1 <> falseOneTrue  b2
        , falseOneTrue  b1 <> falseMoreTrue b2
        , falseMoreTrue b1 <> falseOneTrue  b2
        , falseMoreTrue b1 <> falseMoreTrue b2
        ]
    }


-- | Empty `Best4` for a given tree node.  Think of `mempty` with obligatory
-- vertex and potential argument.
emptyBest4 :: G.Vertex -> Double -> Best4
emptyBest4 node pot = Best4
  { true = addNode node True pot mempty
  , falseZeroTrue = addNode node False 0.0 mempty
  , falseOneTrue = impossible
  , falseMoreTrue = impossible
  }
